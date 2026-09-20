use crate::http_client::sse::BoxedStream;
use bytes::Bytes;
pub use http::{HeaderMap, HeaderValue, Method, Request, Response, Uri, request::Builder};
use http::{HeaderName, StatusCode};
use reqwest::Body;
pub mod multipart;
pub mod retry;
pub mod sse;
use crate::wasm_compat::*;
pub use multipart::MultipartForm;
pub use reqwest::Client as ReqwestClient;
use std::pin::Pin;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Http error: {0}")]
    Protocol(#[from] http::Error),
    #[error("Invalid status code: {0}")]
    InvalidStatusCode(StatusCode),
    #[error("Invalid status code {0} with message: {1}")]
    InvalidStatusCodeWithMessage(StatusCode, String),
    /// Same shape/Display as [`Self::InvalidStatusCodeWithMessage`] — the only
    /// difference is `retry_after`, read from the response's `retry-after`
    /// header. Kept as a **separate** variant (rather than adding a field to
    /// `InvalidStatusCodeWithMessage`) because several call sites across the
    /// crate construct that variant directly with exactly two positional
    /// fields (providers, `test_utils/http.rs`, the `impl_provider_response_helpers!`
    /// macro); giving it a third field would force every one of those sites to
    /// change for a capability only the real `send_streaming` response path
    /// needs. Display is **byte-identical** to `InvalidStatusCodeWithMessage`
    /// (retry_after never renders) so string-matching callers see no change.
    #[error("Invalid status code {status} with message: {body}")]
    InvalidStatusCodeWithHeaders {
        status: StatusCode,
        body: String,
        retry_after: Option<String>,
    },
    #[error("Header value outside of legal range: {0}")]
    InvalidHeaderValue(#[from] http::header::InvalidHeaderValue),
    #[error("Request in error state, cannot access headers")]
    NoHeaders,
    #[error("Stream ended")]
    StreamEnded,
    #[error("Invalid content type was returned: {0:?}")]
    InvalidContentType(HeaderValue),
    #[cfg(not(target_family = "wasm"))]
    #[error("Http client error: {0}")]
    Instance(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),

    #[cfg(target_family = "wasm")]
    #[error("Http client error: {0}")]
    Instance(#[from] Box<dyn std::error::Error + 'static>),
}

impl Error {
    /// The non-success HTTP status this error carries, when it wraps one.
    pub fn non_success_status(&self) -> Option<StatusCode> {
        match self {
            Self::InvalidStatusCode(status)
            | Self::InvalidStatusCodeWithMessage(status, _)
            | Self::InvalidStatusCodeWithHeaders { status, .. } => Some(*status),
            _ => None,
        }
    }

    /// The raw response body this error preserved, when it has one.
    pub fn response_body(&self) -> Option<&str> {
        match self {
            Self::InvalidStatusCodeWithMessage(_, body)
            | Self::InvalidStatusCodeWithHeaders { body, .. } => Some(body.as_str()),
            _ => None,
        }
    }

    /// The `retry-after` response header value, verbatim (not parsed — callers
    /// decide how to interpret seconds vs. an HTTP-date). Only the header-carrying
    /// variant (built from a real `send_streaming` non-success response) has one.
    pub fn retry_after(&self) -> Option<&str> {
        match self {
            Self::InvalidStatusCodeWithHeaders { retry_after, .. } => retry_after.as_deref(),
            _ => None,
        }
    }

    pub(crate) fn non_success_body(&self) -> Option<&str> {
        self.response_body()
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(not(target_family = "wasm"))]
pub(crate) fn instance_error<E: std::error::Error + Send + Sync + 'static>(error: E) -> Error {
    Error::Instance(error.into())
}

#[cfg(target_family = "wasm")]
fn instance_error<E: std::error::Error + 'static>(error: E) -> Error {
    Error::Instance(error.into())
}

async fn non_success_status_error(response: reqwest::Response) -> Error {
    let status = response.status();
    let message = response
        .text()
        .await
        .unwrap_or_else(|error| format!("failed to read error response body: {error}"));
    Error::InvalidStatusCodeWithMessage(status, message)
}

/// Same as [`non_success_status_error`] but also preserves the `retry-after`
/// response header. Used **only** by `send_streaming`'s live non-success path
/// (the one Anthropic — and every other streaming provider — actually hits):
/// the header must be read before `.text()` consumes the response body.
/// `send`/`send_multipart` (`into_lazy_response`) keep calling the plain
/// `non_success_status_error` above unchanged — several provider call sites
/// pattern-match `InvalidStatusCodeWithMessage(status, message)` on errors from
/// those two methods (`client/mod.rs`, `providers/deepseek.rs`,
/// `providers/xiaomimimo.rs`), and widening what they get back is out of scope
/// for this change.
async fn non_success_status_error_with_headers(response: reqwest::Response) -> Error {
    let status = response.status();
    let retry_after = response
        .headers()
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let body = response
        .text()
        .await
        .unwrap_or_else(|error| format!("failed to read error response body: {error}"));
    Error::InvalidStatusCodeWithHeaders { status, body, retry_after }
}

pub type LazyBytes = WasmBoxedFuture<'static, Result<Bytes>>;
pub type LazyBody<T> = WasmBoxedFuture<'static, Result<T>>;

pub type StreamingResponse = Response<BoxedStream>;

#[derive(Debug, Clone, Copy)]
pub struct NoBody;

impl From<NoBody> for Bytes {
    fn from(_: NoBody) -> Self {
        Bytes::new()
    }
}

impl From<NoBody> for Body {
    fn from(_: NoBody) -> Self {
        reqwest::Body::default()
    }
}

pub async fn text(response: Response<LazyBody<Vec<u8>>>) -> Result<String> {
    let text = response.into_body().await?;
    Ok(String::from(String::from_utf8_lossy(&text)))
}

pub fn make_auth_header(key: impl AsRef<str>) -> Result<(HeaderName, HeaderValue)> {
    Ok((
        http::header::AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {}", key.as_ref()))?,
    ))
}

pub fn bearer_auth_header(headers: &mut HeaderMap, key: impl AsRef<str>) -> Result<()> {
    let (k, v) = make_auth_header(key)?;

    headers.insert(k, v);

    Ok(())
}

pub fn with_bearer_auth(mut req: Builder, auth: &str) -> Result<Builder> {
    bearer_auth_header(req.headers_mut().ok_or(Error::NoHeaders)?, auth)?;

    Ok(req)
}

/// A helper trait to make generic requests (both regular and SSE) possible.
pub trait HttpClientExt: WasmCompatSend + WasmCompatSync {
    /// Send a HTTP request, get a response back (as bytes). Response must be able to be turned back into Bytes.
    fn send<T, U>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes>,
        T: WasmCompatSend,
        U: From<Bytes>,
        U: WasmCompatSend + 'static;

    /// Send a HTTP request with a multipart body, get a response back (as bytes). Response must be able to be turned back into Bytes (although usually for the response, you will probably want to specify Bytes anyway).
    fn send_multipart<U>(
        &self,
        req: Request<MultipartForm>,
    ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes>,
        U: WasmCompatSend + 'static;

    /// Send a HTTP request, get a streamed response back (as a stream of [`bytes::Bytes`].)
    fn send_streaming<T>(
        &self,
        req: Request<T>,
    ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend;
}

async fn into_lazy_response<U>(response: reqwest::Response) -> Result<Response<LazyBody<U>>>
where
    U: From<Bytes>,
    U: WasmCompatSend + 'static,
{
    if !response.status().is_success() {
        return Err(non_success_status_error(response).await);
    }

    let mut res = Response::builder().status(response.status());

    if let Some(headers) = res.headers_mut() {
        *headers = response.headers().clone();
    }

    let body: LazyBody<U> = Box::pin(async {
        let bytes = response.bytes().await.map_err(instance_error)?;
        Ok(U::from(bytes))
    });

    res.body(body).map_err(Error::Protocol)
}

macro_rules! impl_http_client_ext {
    ($(#[$attribute:meta])* $client:ty) => {
        $(#[$attribute])*
        impl HttpClientExt for $client {
            fn send<T, U>(
                &self,
                req: Request<T>,
            ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
            where
                T: Into<Bytes>,
                U: From<Bytes> + WasmCompatSend + 'static,
            {
                let (parts, body) = req.into_parts();
                let req = self
                    .request(parts.method, parts.uri.to_string())
                    .headers(parts.headers)
                    .body(body.into());

                async move {
                    let response = req.send().await.map_err(instance_error)?;
                    into_lazy_response(response).await
                }
            }

            fn send_multipart<U>(
                &self,
                req: Request<MultipartForm>,
            ) -> impl Future<Output = Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
            where
                U: From<Bytes>,
                U: WasmCompatSend + 'static,
            {
                let (parts, body) = req.into_parts();
                let body = reqwest::multipart::Form::from(body);

                let req = self
                    .request(parts.method, parts.uri.to_string())
                    .headers(parts.headers)
                    .multipart(body);

                async move {
                    let response = req.send().await.map_err(instance_error)?;
                    into_lazy_response(response).await
                }
            }

            fn send_streaming<T>(
                &self,
                req: Request<T>,
            ) -> impl Future<Output = Result<StreamingResponse>> + WasmCompatSend
            where
                T: Into<Bytes> + WasmCompatSend,
            {
                let (parts, body) = req.into_parts();

                let client = self.clone();

                async move {
                    let req = self
                        .request(parts.method, parts.uri.to_string())
                        .headers(parts.headers)
                        .body(body.into())
                        .build()
                        .map_err(|error| Error::Instance(error.into()))?;
                    let response: reqwest::Response =
                        client.execute(req).await.map_err(instance_error)?;
                    if !response.status().is_success() {
                        return Err(non_success_status_error_with_headers(response).await);
                    }

                    #[cfg(not(target_family = "wasm"))]
                    let mut res = Response::builder()
                        .status(response.status())
                        .version(response.version());

                    #[cfg(target_family = "wasm")]
                    let mut res = Response::builder().status(response.status());

                    if let Some(hs) = res.headers_mut() {
                        *hs = response.headers().clone();
                    }

                    use futures::StreamExt;

                    let mapped_stream: Pin<
                        Box<dyn WasmCompatSendStream<InnerItem = Result<Bytes>>>,
                    > = Box::pin(
                        response
                            .bytes_stream()
                            .map(|chunk| chunk.map_err(|e| Error::Instance(Box::new(e)))),
                    );

                    res.body(mapped_stream).map_err(Error::Protocol)
                }
            }
        }
    };
}

impl_http_client_ext!(reqwest::Client);

impl_http_client_ext!(
    #[cfg(feature = "reqwest-middleware")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest-middleware")))]
    reqwest_middleware::ClientWithMiddleware
);

#[cfg(test)]
mod tests {
    use super::*;

    /// The new header-carrying variant must render the **same** Display text as
    /// the pre-existing `InvalidStatusCodeWithMessage(status, body)` — callers
    /// (xyrend's `LlmCallError::parse`, downstream provider error strings) match
    /// on that exact string and must not see it change shape when retry-after
    /// happens to be present.
    #[test]
    fn invalid_status_code_with_headers_display_matches_with_message() {
        let a = Error::InvalidStatusCodeWithMessage(
            StatusCode::TOO_MANY_REQUESTS,
            "slow down".to_string(),
        );
        let b = Error::InvalidStatusCodeWithHeaders {
            status: StatusCode::TOO_MANY_REQUESTS,
            body: "slow down".to_string(),
            retry_after: Some("7".to_string()),
        };
        assert_eq!(a.to_string(), b.to_string());
        assert_eq!(
            b.to_string(),
            "Invalid status code 429 Too Many Requests with message: slow down"
        );
    }

    #[test]
    fn invalid_status_code_with_headers_exposes_status_body_and_retry_after() {
        let e = Error::InvalidStatusCodeWithHeaders {
            status: StatusCode::SERVICE_UNAVAILABLE,
            body: "boom".to_string(),
            retry_after: Some("30".to_string()),
        };
        assert_eq!(e.non_success_status(), Some(StatusCode::SERVICE_UNAVAILABLE));
        assert_eq!(e.response_body(), Some("boom"));
        assert_eq!(e.retry_after(), Some("30"));
    }

    /// The plain (no-headers) variant still answers `non_success_status`/`response_body`
    /// (unchanged behaviour) but never carries a retry-after — the accessor is
    /// specific to the header-carrying variant.
    #[test]
    fn invalid_status_code_with_message_has_no_retry_after() {
        let e = Error::InvalidStatusCodeWithMessage(StatusCode::BAD_REQUEST, "x".to_string());
        assert_eq!(e.non_success_status(), Some(StatusCode::BAD_REQUEST));
        assert_eq!(e.response_body(), Some("x"));
        assert_eq!(e.retry_after(), None);
    }

    /// `InvalidStatusCode` (status only, no body) still yields a status but no
    /// body/retry-after via the newly-public accessors.
    #[test]
    fn invalid_status_code_bare_has_no_body_or_retry_after() {
        let e = Error::InvalidStatusCode(StatusCode::NOT_FOUND);
        assert_eq!(e.non_success_status(), Some(StatusCode::NOT_FOUND));
        assert_eq!(e.response_body(), None);
        assert_eq!(e.retry_after(), None);
    }

    #[test]
    fn unrelated_variant_has_no_status_body_or_retry_after() {
        let e = Error::StreamEnded;
        assert_eq!(e.non_success_status(), None);
        assert_eq!(e.response_body(), None);
        assert_eq!(e.retry_after(), None);
    }
}
