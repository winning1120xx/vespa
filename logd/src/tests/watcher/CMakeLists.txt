# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
vespa_add_executable(logd_watcher_test_app TEST
    SOURCES
    watcher_test.cpp
    DEPENDS
    logd
    GTest::gtest
)
vespa_add_test(NAME logd_watcher_test_app COMMAND logd_watcher_test_app)
