# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
vespa_add_executable(searchlib_document_store_test_app TEST
    SOURCES
    document_store_test.cpp
    DEPENDS
    vespa_searchlib
    GTest::gtest
)
vespa_add_test(NAME searchlib_document_store_test_app COMMAND searchlib_document_store_test_app)
vespa_add_executable(searchlib_visitcache_test_app TEST
    SOURCES
    visitcache_test.cpp
    DEPENDS
    vespa_searchlib
    GTest::gtest
)
vespa_add_test(NAME searchlib_visitcache_test_app COMMAND searchlib_visitcache_test_app)
