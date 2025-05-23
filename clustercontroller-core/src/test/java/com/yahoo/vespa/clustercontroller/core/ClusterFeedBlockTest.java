// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.clustercontroller.core;

import com.yahoo.vdslib.state.Node;
import com.yahoo.vdslib.state.NodeState;
import com.yahoo.vdslib.state.NodeType;
import com.yahoo.vdslib.state.State;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.yahoo.vespa.clustercontroller.core.FeedBlockUtil.createResourceUsageJson;
import static com.yahoo.vespa.clustercontroller.core.FeedBlockUtil.mapOf;
import static com.yahoo.vespa.clustercontroller.core.FeedBlockUtil.setOf;
import static com.yahoo.vespa.clustercontroller.core.FeedBlockUtil.usage;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

@ExtendWith(CleanupZookeeperLogsOnSuccess.class)
public class ClusterFeedBlockTest extends FleetControllerTest {

    private static final int NODE_COUNT = 3;

    private final Timer timer = new FakeTimer();

    private FleetController ctrl;
    private DummyCommunicator communicator;

    private void initialize(FleetControllerOptions.Builder builder) throws Exception {
        List<Node> nodes = new ArrayList<>();
        for (int i = 0; i < builder.nodes().size(); ++i) {
            nodes.add(new Node(NodeType.STORAGE, i));
            nodes.add(new Node(NodeType.DISTRIBUTOR, i));
        }

        communicator = new DummyCommunicator(nodes, timer);
        setUpZooKeeperServer(builder);
        options = builder.build();
        var context = new TestFleetControllerContext(options);
        boolean start = false;
        ctrl = createFleetController(timer, options, context, communicator, communicator, null, start);

        ctrl.tick();
        markAllNodesAsUp(options);
        ctrl.tick();
    }

    private void markAllNodesAsUp(FleetControllerOptions options) throws Exception {
        for (int i = 0; i < options.nodes().size(); ++i) {
            communicator.setNodeState(new Node(NodeType.STORAGE, i), State.UP, "");
            communicator.setNodeState(new Node(NodeType.DISTRIBUTOR, i), State.UP, "");
        }
        ctrl.tick();
    }

    private static FleetControllerOptions.Builder createOptions(Map<String, Double> feedBlockLimits, double clusterFeedBlockNoiseLevel) {
        return defaultOptions()
                .setDistributionConfig(DistributionBuilder.configForFlatCluster(NODE_COUNT))
                .setNodes(new HashSet<>(DistributionBuilder.buildConfiguredNodes(NODE_COUNT)))
                .setClusterFeedBlockEnabled(true)
                .setClusterFeedBlockLimit(feedBlockLimits)
                .setClusterFeedBlockNoiseLevel(clusterFeedBlockNoiseLevel);
    }

    private static FleetControllerOptions.Builder createOptions(Map<String, Double> feedBlockLimits) {
        return createOptions(feedBlockLimits, 0.0);
    }

    private void reportResourceUsageFromNode(int nodeIndex, State nodeState, Set<FeedBlockUtil.UsageDetails> resourceUsages) throws Exception {
        String hostInfo = createResourceUsageJson(resourceUsages);
        communicator.setNodeState(new Node(NodeType.STORAGE, nodeIndex), new NodeState(NodeType.STORAGE, nodeState), hostInfo);
        ctrl.tick();
    }

    private void reportResourceUsageFromNode(int nodeIndex, Set<FeedBlockUtil.UsageDetails> resourceUsages) throws Exception {
        reportResourceUsageFromNode(nodeIndex, State.UP, resourceUsages);
    }

    @Test
    void cluster_feed_can_be_blocked_and_unblocked_by_single_node() throws Exception {
        initialize(createOptions(mapOf(usage("cheese", 0.7), usage("wine", 0.4))));
        assertFalse(ctrl.getClusterStateBundle().clusterFeedIsBlocked());

        // Too much cheese in use, must block feed!
        reportResourceUsageFromNode(1, setOf(usage("cheese", 0.8), usage("wine", 0.3)));
        assertTrue(ctrl.getClusterStateBundle().clusterFeedIsBlocked());
        // TODO check desc?

        // Wine usage has gone up too, we should remain blocked
        reportResourceUsageFromNode(1, setOf(usage("cheese", 0.8), usage("wine", 0.5)));
        assertTrue(ctrl.getClusterStateBundle().clusterFeedIsBlocked());
        // TODO check desc?

        // Back to normal wine and cheese levels
        reportResourceUsageFromNode(1, setOf(usage("cheese", 0.6), usage("wine", 0.3)));
        assertFalse(ctrl.getClusterStateBundle().clusterFeedIsBlocked());
    }

    @Test
    void cluster_feed_block_state_is_recomputed_when_options_are_updated() throws Exception {
        initialize(createOptions(mapOf(usage("cheese", 0.7), usage("wine", 0.4))));
        assertFalse(ctrl.getClusterStateBundle().clusterFeedIsBlocked());

        reportResourceUsageFromNode(1, setOf(usage("cheese", 0.8), usage("wine", 0.3)));
        assertTrue(ctrl.getClusterStateBundle().clusterFeedIsBlocked());

        // Increase cheese allowance. Should now automatically unblock since reported usage is lower.
        ctrl.updateOptions(createOptions(mapOf(usage("cheese", 0.9), usage("wine", 0.4))).build());
        ctrl.tick(); // Options propagation
        ctrl.tick(); // State recomputation
        assertFalse(ctrl.getClusterStateBundle().clusterFeedIsBlocked());
    }

    private String decorate(String msg) {
        return ResourceExhaustionCalculator.decoratedMessage(ctrl.getCluster(), msg);
    }

    @Test
    void cluster_feed_block_state_is_recomputed_when_resource_block_set_differs() throws Exception {
        initialize(createOptions(mapOf(usage("cheese", 0.7), usage("wine", 0.4))));
        assertFalse(ctrl.getClusterStateBundle().clusterFeedIsBlocked());

        reportResourceUsageFromNode(1, setOf(usage("cheese", 0.8), usage("wine", 0.3)));
        var bundle = ctrl.getClusterStateBundle();
        assertTrue(bundle.clusterFeedIsBlocked());
        assertEquals(decorate("cheese on node 1 [unknown hostname] is 80.0% full (the configured limit is 70.0%)"),
                     bundle.getFeedBlock().get().getDescription());

        reportResourceUsageFromNode(1, setOf(usage("cheese", 0.8), usage("wine", 0.5)));
        bundle = ctrl.getClusterStateBundle();
        assertTrue(bundle.clusterFeedIsBlocked());
        assertEquals(decorate("cheese on node 1 [unknown hostname] is 80.0% full (the configured limit is 70.0%), " +
                              "wine on node 1 [unknown hostname] is 50.0% full (the configured limit is 40.0%)"),
                     bundle.getFeedBlock().get().getDescription());
    }

    @Test
    void cluster_feed_block_state_is_not_recomputed_when_only_resource_usage_levels_differ() throws Exception {
        initialize(createOptions(mapOf(usage("cheese", 0.7), usage("wine", 0.4))));
        assertFalse(ctrl.getClusterStateBundle().clusterFeedIsBlocked());

        reportResourceUsageFromNode(1, setOf(usage("cheese", 0.8), usage("wine", 0.3)));
        var bundle = ctrl.getClusterStateBundle();
        assertTrue(bundle.clusterFeedIsBlocked());
        assertEquals(decorate("cheese on node 1 [unknown hostname] is 80.0% full (the configured limit is 70.0%)"),
                     bundle.getFeedBlock().get().getDescription());

        // 80% -> 90%, should not trigger new state.
        reportResourceUsageFromNode(1, setOf(usage("cheese", 0.9), usage("wine", 0.3)));
        bundle = ctrl.getClusterStateBundle();
        assertTrue(bundle.clusterFeedIsBlocked());
        assertEquals(decorate("cheese on node 1 [unknown hostname] is 80.0% full (the configured limit is 70.0%)"),
                     bundle.getFeedBlock().get().getDescription());
    }

    @Test
    void cluster_feed_block_state_is_recomputed_when_usage_enters_hysteresis_range() throws Exception {
        initialize(createOptions(mapOf(usage("cheese", 0.7), usage("wine", 0.4)), 0.1));
        assertFalse(ctrl.getClusterStateBundle().clusterFeedIsBlocked());

        reportResourceUsageFromNode(1, setOf(usage("cheese", 0.75), usage("wine", 0.3)));
        var bundle = ctrl.getClusterStateBundle();
        assertTrue(bundle.clusterFeedIsBlocked());
        assertEquals(decorate("cheese on node 1 [unknown hostname] is 75.0% full (the configured limit is 70.0%)"),
                     bundle.getFeedBlock().get().getDescription());

        reportResourceUsageFromNode(1, setOf(usage("cheese", 0.68), usage("wine", 0.3)));
        bundle = ctrl.getClusterStateBundle();
        assertTrue(bundle.clusterFeedIsBlocked());
        // FIXME Effective limit is modified by hysteresis but due to how we check state deltas this
        // is not discovered here. Still correct in terms of what resources are blocked or not, but
        // the description is not up to date here.
        assertEquals(decorate("cheese on node 1 [unknown hostname] is 75.0% full (the configured limit is 70.0%)"),
                     bundle.getFeedBlock().get().getDescription());

        // Trigger an explicit recompute by adding a separate resource exhaustion
        reportResourceUsageFromNode(1, setOf(usage("cheese", 0.67), usage("wine", 0.5)));
        bundle = ctrl.getClusterStateBundle();
        assertTrue(bundle.clusterFeedIsBlocked());
        assertEquals(decorate("cheese on node 1 [unknown hostname] is 67.0% full (the configured limit is 70.0%, effective limit lowered to 60.0% until feed unblocked), " +
                              "wine on node 1 [unknown hostname] is 50.0% full (the configured limit is 40.0%)"), // Not under hysteresis
                     bundle.getFeedBlock().get().getDescription());

        // Wine usage drops beyond hysteresis range, should be unblocked immediately.
        reportResourceUsageFromNode(1, setOf(usage("cheese", 0.611), usage("wine", 0.2)));
        bundle = ctrl.getClusterStateBundle();
        assertTrue(bundle.clusterFeedIsBlocked());
        assertEquals(decorate("cheese on node 1 [unknown hostname] is 61.1% full (the configured limit is 70.0%, effective limit lowered to 60.0% until feed unblocked)"),
                     bundle.getFeedBlock().get().getDescription());

        // Cheese now drops below hysteresis range, should be unblocked as well.
        reportResourceUsageFromNode(1, setOf(usage("cheese", 0.59), usage("wine", 0.2)));
        bundle = ctrl.getClusterStateBundle();
        assertFalse(bundle.clusterFeedIsBlocked());
    }

    @Test
    void unavailable_nodes_are_not_considered_when_computing_feed_blocked_state() throws Exception {
        initialize(createOptions(mapOf(usage("cheese", 0.7), usage("wine", 0.4)), 0.1));
        assertFalse(ctrl.getClusterStateBundle().clusterFeedIsBlocked());

        reportResourceUsageFromNode(1, State.DOWN, setOf(usage("cheese", 0.8), usage("wine", 0.5)));
        // Not blocked, node with exhaustion is marked as Down
        assertFalse(ctrl.getClusterStateBundle().clusterFeedIsBlocked());
    }

    // FIXME implicit changes in limits due to hysteresis adds spurious exhaustion remove+add node event pair

}
