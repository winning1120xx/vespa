// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.jdisc.http.server.jetty;

import ai.vespa.utils.BytesQuantity;
import com.google.inject.Inject;
import com.yahoo.component.provider.ComponentRegistry;
import com.yahoo.container.logging.ConnectionLog;
import com.yahoo.container.logging.RequestLog;
import com.yahoo.jdisc.AbstractResource;
import com.yahoo.jdisc.Metric;
import com.yahoo.jdisc.http.ConnectorConfig;
import com.yahoo.jdisc.http.ServerConfig;
import com.yahoo.jdisc.service.CurrentContainer;
import com.yahoo.jdisc.service.ServerProvider;
import org.eclipse.jetty.jmx.ConnectorServer;
import org.eclipse.jetty.jmx.MBeanContainer;
import org.eclipse.jetty.server.Connector;
import org.eclipse.jetty.server.Handler;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.server.ServerConnector;
import org.eclipse.jetty.server.SslConnectionFactory;
import org.eclipse.jetty.server.handler.ContextHandlerCollection;
import org.eclipse.jetty.server.handler.ErrorHandler;
import org.eclipse.jetty.server.handler.StatisticsHandler;
import org.eclipse.jetty.server.handler.gzip.GzipHandler;
import org.eclipse.jetty.util.thread.QueuedThreadPool;

import javax.management.remote.JMXServiceURL;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.net.BindException;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * @author Simon Thoresen Hult
 * @author bjorncs
 */
public class JettyHttpServer extends AbstractResource implements ServerProvider {

    private final static Logger log = Logger.getLogger(JettyHttpServer.class.getName());

    private final ServerConfig config;
    private final Server server;
    private final List<Integer> listenedPorts = new ArrayList<>();
    private final ServerMetricReporter metricsReporter;
    private final Deque<JDiscContext> contexts = new ConcurrentLinkedDeque<>();

    @Inject // ServerProvider implementors must use com.google.inject.Inject
    public JettyHttpServer(Metric metric,
                           ServerConfig serverConfig,
                           ComponentRegistry<ConnectorFactory> connectorFactories,
                           RequestLog requestLog,
                           ConnectionLog connectionLog) {
        if (connectorFactories.allComponents().isEmpty())
            throw new IllegalArgumentException("No connectors configured.");

        this.config = serverConfig;
        server = new Server();

        // Create a custom error handler
        // - Increased buffer size to avoid buffer overflow for large error messages (e.g. massive YQL query in URI).
        // - Show stack trace and cause when developer mode is enabled.
        var errorHandler = new ErrorHandler();
        errorHandler.setBufferSize((int)BytesQuantity.ofKB(16).toBytes());
        errorHandler.setShowStacks(serverConfig.developerMode());
        errorHandler.setShowCauses(serverConfig.developerMode());
        server.setErrorHandler(errorHandler);

        server.setStopTimeout((long)(serverConfig.stopTimeout() * 1000.0));
        var metricAggregatingRequestLog = new MetricAggregatingRequestLog(config.metric());
        server.addBean(metricAggregatingRequestLog);
        if (requestLog instanceof VoidRequestLog) {
            server.setRequestLog(metricAggregatingRequestLog);
        } else {
            server.setRequestLog(new org.eclipse.jetty.server.RequestLog.Collection(
                    new AccessLogRequestLog(requestLog),
                    metricAggregatingRequestLog));
        }
        setupJmx(server, serverConfig);
        configureJettyThreadpool(server, serverConfig);

        var perConnectorHandlers = new ContextHandlerCollection();
        for (ConnectorFactory connectorFactory : connectorFactories.allComponents()) {
            ConnectorConfig connectorConfig = connectorFactory.getConnectorConfig();
            var connector = connectorFactory.createConnector(metric, server);
            server.addConnector(connector);
            listenedPorts.add(connectorConfig.listenPort());
            var connectorCfg = connector.connectorConfig();
            var jdiscHandler = new JdiscDispatchingHandler(this::newestContext);
            var authEnforcerEnabled = connectorCfg.tlsClientAuthEnforcer().enable();
            var contextHandler = new ConnectorSpecificContextHandler(
                    connector,
                    authEnforcerEnabled ? newTlsClientAuthEnforcerHandler(connectorCfg, jdiscHandler) : jdiscHandler);
            server.addBean(contextHandler);
            perConnectorHandlers.addHandler(contextHandler);
        }

        var statisticsHandler = new StatisticsHandler(newGzipHandler(perConnectorHandlers));
        var connectionMetricAggregator = new ConnectionMetricAggregator(serverConfig, metric, statisticsHandler);


        if (!(connectionLog instanceof VoidConnectionLog)) {
            var connectionLogger = new JettyConnectionLogger(serverConfig.connectionLog(), connectionLog, connectionMetricAggregator);
            server.addBeanToAllConnectors(connectionLogger);
            server.setHandler(connectionLogger);
        } else {
            server.setHandler(connectionMetricAggregator);
        }

        server.addBeanToAllConnectors(connectionMetricAggregator);

        this.metricsReporter = new ServerMetricReporter(metric, server, statisticsHandler, metricAggregatingRequestLog);
    }

    JDiscContext registerContext(FilterBindings filterBindings, CurrentContainer container, Janitor janitor, Metric metric) {
        JDiscContext context = JDiscContext.of(filterBindings, container, janitor, metric, config);
        contexts.addFirst(context);
        return context;
    }

    void deregisterContext(JDiscContext context) {
        contexts.remove(context);
    }

    JDiscContext newestContext() {
        JDiscContext context = contexts.peekFirst();
        if (context == null) throw new IllegalStateException("JettyHttpServer has no registered JDiscContext");
        return context;
    }

    private static void setupJmx(Server server, ServerConfig serverConfig) {
        if (serverConfig.jmx().enabled()) {
            System.setProperty("java.rmi.server.hostname", "localhost");
            server.addBean(new MBeanContainer(ManagementFactory.getPlatformMBeanServer()));
            server.addBean(new ConnectorServer(createJmxLoopbackOnlyServiceUrl(serverConfig.jmx().listenPort()),
                                               "org.eclipse.jetty.jmx:name=rmiconnectorserver"));
        }
    }

    private static void configureJettyThreadpool(Server server, ServerConfig config) {
        int cpus = Runtime.getRuntime().availableProcessors();
        QueuedThreadPool pool = (QueuedThreadPool) server.getThreadPool();
        int maxThreads = config.maxWorkerThreads() > 0 ? config.maxWorkerThreads() : 16 + cpus;
        pool.setMaxThreads(maxThreads);
        int minThreads = config.minWorkerThreads() >= 0 ? config.minWorkerThreads() : 16 + cpus;
        pool.setMinThreads(minThreads);
        log.info(String.format("Threadpool size: min=%d, max=%d", minThreads, maxThreads));
    }

    private static JMXServiceURL createJmxLoopbackOnlyServiceUrl(int port) {
        try {
            return new JMXServiceURL("rmi", "localhost", port, "/jndi/rmi://localhost:" + port + "/jmxrmi");
        } catch (MalformedURLException e) {
            throw new RuntimeException(e);
        }
    }

    private static String getDisplayName(List<Integer> ports) {
        return ports.stream().map(Object::toString).collect(Collectors.joining(":"));
    }

    @Override
    public void start() {
        try {
            server.start();
            if (config.metric().reporterEnabled()) metricsReporter.start();
            logEffectiveSslConfiguration();
        } catch (final Exception e) {
            if (e instanceof IOException && e.getCause() instanceof BindException) {
                throw new RuntimeException("Failed to start server due to BindException. ListenPorts = " + listenedPorts.toString(), e.getCause());
            }
            throw new RuntimeException("Failed to start server.", e);
        }
    }

    private void logEffectiveSslConfiguration() {
        if (!server.isStarted()) throw new IllegalStateException();
        for (Connector connector : server.getConnectors()) {
            ServerConnector serverConnector = (ServerConnector) connector;
            int localPort = serverConnector.getLocalPort();
            var sslConnectionFactory = serverConnector.getConnectionFactory(SslConnectionFactory.class);
            if (sslConnectionFactory != null) {
                var sslContextFactory = sslConnectionFactory.getSslContextFactory();
                String protocols = Arrays.toString(sslContextFactory.getSelectedProtocols());
                String cipherSuites = Arrays.toString(sslContextFactory.getSelectedCipherSuites());
                log.info(String.format("TLS for port '%d': %s with %s", localPort, protocols, cipherSuites));
            }
        }
    }

    @Override
    public void close() {
        try {
            log.log(Level.INFO, String.format("Shutting down Jetty server (graceful=%b, timeout=%.1fs)",
                    isGracefulShutdownEnabled(), server.getStopTimeout()/1000d));
            long start = System.currentTimeMillis();
            server.stop();
            log.log(Level.INFO, String.format("Jetty server shutdown completed in %.3f seconds",
                    (System.currentTimeMillis()-start)/1000D));
        } catch (final Exception e) {
            log.log(Level.SEVERE, "Jetty server shutdown threw an unexpected exception.", e);
        }

        metricsReporter.shutdown();
    }

    private boolean isGracefulShutdownEnabled() { return server.getStopTimeout() > 0; }

    public int getListenPort() {
        return ((ServerConnector)server.getConnectors()[0]).getLocalPort();
    }

    Server server() { return server; }

    private static TlsClientAuthenticationEnforcer newTlsClientAuthEnforcerHandler(ConnectorConfig cfg, Handler handler) {
        return new TlsClientAuthenticationEnforcer(cfg.tlsClientAuthEnforcer(), handler);
    }

    private static GzipHandler newGzipHandler(Handler handler) {
        var h = new GzipHandler(handler);
        h.setInflateBufferSize(8 * 1024);
        h.setIncludedMethods("GET", "POST", "PUT", "PATCH");
        return h;
    }
}
