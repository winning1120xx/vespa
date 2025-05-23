// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.jdisc.http.server.jetty;

import com.google.common.base.Objects;
import com.yahoo.container.logging.AccessLogEntry;
import com.yahoo.container.logging.RequestLog;
import com.yahoo.container.logging.RequestLogEntry;
import com.yahoo.jdisc.http.HttpRequest;
import org.eclipse.jetty.http.HttpVersion;
import org.eclipse.jetty.io.EndPoint;
import org.eclipse.jetty.server.Request;
import org.eclipse.jetty.server.Response;
import org.eclipse.jetty.util.component.AbstractLifeCycle;

import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.OptionalInt;
import java.util.UUID;
import java.util.function.BiConsumer;
import java.util.logging.Level;
import java.util.logging.Logger;

import static com.yahoo.jdisc.http.server.jetty.RequestUtils.getConnector;
import static com.yahoo.jdisc.http.server.jetty.RequestUtils.getConnectorLocalPort;

/**
 * Implements the request accessing logging for Jetty.
 *
 * @author Oyvind Bakksjo
 * @author bjorncs
 */
class AccessLogRequestLog extends AbstractLifeCycle implements org.eclipse.jetty.server.RequestLog {

    private static final Logger logger = Logger.getLogger(AccessLogRequestLog.class.getName());

    // HTTP headers that are logged as extra key-value-pairs in access log entries
    private static final List<String> LOGGED_REQUEST_HEADERS = List.of("Vespa-Client-Version");

    private final RequestLog requestLog;

    AccessLogRequestLog(RequestLog requestLog) {
        this.requestLog = requestLog;
    }

    @Override
    public void log(Request request, Response response) {
        try {
            RequestLogEntry.Builder builder = new RequestLogEntry.Builder();

            String peerAddress = Request.getRemoteAddr(request);
            int peerPort = Request.getRemotePort(request);
            long startTime = Request.getTimeStamp(request);
            long endTime = System.currentTimeMillis();
            Integer statusCodeOverride = (Integer) request.getAttribute(JdiscDispatchingHandler.ACCESS_LOG_STATUS_CODE_OVERRIDE);
            builder.peerAddress(peerAddress)
                    .peerPort(peerPort)
                    .localPort(getLocalPort(request))
                    .timestamp(Instant.ofEpochMilli(startTime))
                    .duration(Duration.ofMillis(Math.max(0, endTime - startTime)))
                    .responseSize(Response.getContentBytesWritten(response))
                    .requestSize(Request.getContentBytesRead(request))
                    .statusCode(statusCodeOverride != null ? statusCodeOverride : response.getStatus());


            addNonNullValue(builder, request.getMethod(), RequestLogEntry.Builder::httpMethod);
            addNonNullValue(builder, request.getHttpURI().getPath(), RequestLogEntry.Builder::rawPath);
            addNonNullValue(builder, request.getConnectionMetaData().getProtocol(), RequestLogEntry.Builder::httpVersion);
            addNonNullValue(builder, request.getHttpURI().getScheme(), RequestLogEntry.Builder::scheme);
            addNonNullValue(builder, request.getHeaders().get("User-Agent"), RequestLogEntry.Builder::userAgent);
            addNonNullValue(builder, getServerName(request), RequestLogEntry.Builder::hostString);
            addNonNullValue(builder, request.getHeaders().get("Referer"), RequestLogEntry.Builder::referer);
            addNonNullValue(builder, request.getHttpURI().getQuery(), RequestLogEntry.Builder::rawQuery);

            HttpRequest jdiscRequest  = (HttpRequest) request.getAttribute(HttpRequest.class.getName());
            if (jdiscRequest != null) {
                addNonNullValue(builder, jdiscRequest.getUserPrincipal(), RequestLogEntry.Builder::userPrincipal);
            }

            String requestFilterId = (String) request.getAttribute(RequestUtils.JDISC_REQUEST_CHAIN);
            addNonNullValue(builder, requestFilterId, (b, chain) -> b.addExtraAttribute("request-chain", chain));

            String responseFilterId = (String) request.getAttribute(RequestUtils.JDISC_RESPONSE_CHAIN);
            addNonNullValue(builder, responseFilterId, (b, chain) -> b.addExtraAttribute("response-chain", chain));

            UUID connectionId = (UUID) request.getAttribute(JettyConnectionLogger.CONNECTION_ID_REQUEST_ATTRIBUTE);
            addNonNullValue(builder, connectionId, (b, uuid) -> b.connectionId(uuid.toString()));

            String remoteAddress = getRemoteAddress(request);
            if (!Objects.equal(remoteAddress, peerAddress)) {
                builder.remoteAddress(remoteAddress);
            }
            int remotePort = getRemotePort(request);
            if (remotePort != peerPort) {
                builder.remotePort(remotePort);
            }
            LOGGED_REQUEST_HEADERS.forEach(header -> {
                String value = request.getHeaders().get(header);
                if (value != null) {
                    builder.addExtraAttribute(header, value);
                }
            });
            var sslSessionData = (EndPoint.SslSessionData) request.getAttribute(EndPoint.SslSessionData.ATTRIBUTE);
            if (sslSessionData != null) {
                var clientCert = sslSessionData.peerCertificates();
                if (clientCert != null && clientCert.length > 0) {
                    builder.sslPrincipal(clientCert[0].getSubjectX500Principal());
                }
            }

            AccessLogEntry accessLogEntry = (AccessLogEntry) request.getAttribute(JdiscDispatchingHandler.ATTRIBUTE_NAME_ACCESS_LOG_ENTRY);
            if (accessLogEntry != null) {
                var extraAttributes = accessLogEntry.getKeyValues();
                if (extraAttributes != null) {
                    extraAttributes.forEach(builder::addExtraAttributes);
                }
                addNonNullValue(builder, accessLogEntry.getHitCounts(), RequestLogEntry.Builder::hitCounts);
                addNonNullValue(builder, accessLogEntry.getTrace(), RequestLogEntry.Builder::traceNode);
                accessLogEntry.getContent().ifPresent(builder::content);
            }
            http2StreamId(request).ifPresent(streamId -> builder.addExtraAttribute("http2-stream-id", Integer.toString(streamId)));

            requestLog.log(builder.build());
        } catch (Exception e) {
            // Catching any exceptions here as it is unclear how Jetty handles exceptions from a RequestLog.
            logger.log(Level.SEVERE, "Failed to log access log entry: " + e.getMessage(), e);
        }
    }

    private static String getServerName(Request request) {
        try {
            return Request.getServerName(request);
        } catch (IllegalArgumentException e) {
            /*
             * getServerName() may throw IllegalArgumentException for invalid requests where request line contains a URI with relative path.
             * Jetty correctly responds with '400 Bad Request' prior to invoking our request log implementation.
             */
            logger.log(Level.FINE, e, () -> "Fallback to 'Host' header");
            return request.getHeaders().get("Host");
        }
    }

    private String getRemoteAddress(Request request) {
        for (String header : getConnector(request).connectorConfig().accessLog().remoteAddressHeaders()) {
            String value = request.getHeaders().get(header);
            if (value != null) return value;
        }
        return Request.getRemoteAddr(request);
    }

    private int getRemotePort(Request request) {
        for (String header : getConnector(request).connectorConfig().accessLog().remotePortHeaders()) {
            String value = request.getHeaders().get(header);
            if (value != null) {
                OptionalInt maybePort = parsePort(value);
                if (maybePort.isPresent()) return maybePort.getAsInt();
            }
        }
        return Request.getRemotePort(request);
    }

    private static int getLocalPort(Request request) {
        int connectorLocalPort = getConnectorLocalPort(request);
        if (connectorLocalPort <= 0) return Request.getLocalPort(request); // If connector is already closed
        return connectorLocalPort;
    }

    private static OptionalInt parsePort(String port) {
        try {
            return OptionalInt.of(Integer.parseInt(port));
        } catch (IllegalArgumentException e) {
            return OptionalInt.empty();
        }
    }

    private static OptionalInt http2StreamId(Request request) {
        if (request.getConnectionMetaData().getHttpVersion() != HttpVersion.HTTP_2) return OptionalInt.empty();
        return OptionalInt.of(Integer.parseInt(request.getId()));
    }

    private static <T> void addNonNullValue(
            RequestLogEntry.Builder builder, T value, BiConsumer<RequestLogEntry.Builder, T> setter) {
        if (value != null) {
            setter.accept(builder, value);
        }
    }

}
