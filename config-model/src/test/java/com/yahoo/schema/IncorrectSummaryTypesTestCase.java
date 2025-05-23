// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.schema;

import com.yahoo.schema.parser.ParseException;
import com.yahoo.yolean.Exceptions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Tests importing a search definition with conflicting summary types
 *
 * @author bratseth
 */
public class IncorrectSummaryTypesTestCase extends AbstractSchemaTestCase {
    @Test
    void testImportingIncorrect() throws ParseException {
        try {
            ApplicationBuilder.createFromString(
                    "search incorrectsummarytypes {\n" +
                            "  document incorrectsummarytypes {\n" +
                            "    field somestring type string {\n" +
                            "      indexing: summary\n" +
                            "    }\n" +
                            "  }\n" +
                            "  document-summary incorrect {\n" +
                            "    summary somestring type int {\n" +
                            "    }\n" +
                            "  }\n" +
                            "}\n");
            fail("processing should have failed");
        } catch (RuntimeException e) {
            assertEquals("summary 'somestring' in field 'somestring' is inconsistent with summary 'somestring' " +
                         "in document-summary 'incorrect' in schema 'incorrectsummarytypes': " +
                         "All declarations of the same summary field must have the same type", Exceptions.toMessageString(e));
        }
    }

}
