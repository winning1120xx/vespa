# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
vespa_add_executable(vespalib_sharded_hash_map_test_app
    SOURCES
    sharded_hash_map_test.cpp
    DEPENDS
    vespalib
    GTest::gtest
)
vespa_add_test(NAME vespalib_sharded_hash_map_test_app COMMAND vespalib_sharded_hash_map_test_app)
