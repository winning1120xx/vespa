# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
vespa_add_executable(document_document_type_repo_factory_test_app TEST
    SOURCES
    document_type_repo_factory_test.cpp
    DEPENDS
    vespa_document
    GTest::gtest
)
vespa_add_test(NAME document_document_type_repo_factory_test_app COMMAND document_document_type_repo_factory_test_app)
