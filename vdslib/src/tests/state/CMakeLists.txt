# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
vespa_add_library(vdslib_teststate
    SOURCES
    cluster_state_bundle_test.cpp
    clusterstatetest.cpp
    grouptest.cpp
    nodestatetest.cpp
    DEPENDS
    vespa_vdslib
    GTest::gtest
)
