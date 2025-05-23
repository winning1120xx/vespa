// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.indexinglanguage.expressions;

import com.yahoo.document.DataType;

import java.io.PrintStream;

/**
 * @author Simon Thoresen Hult
 */
public final class EchoExpression extends Expression {

    private final PrintStream out;

    public EchoExpression() {
        this(System.out);
    }

    public EchoExpression(PrintStream out) {
        this.out = out;
    }

    @Override
    public boolean isMutating() { return false; }

    public PrintStream getOutputStream() { return out; }

    @Override
    protected void doExecute(ExecutionContext context) {
        out.println(context.getCurrentValue());
    }

    @Override
    public String toString() { return "echo"; }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof EchoExpression rhs)) return false;
        return out == rhs.out;
    }

    @Override
    public int hashCode() {
        return getClass().hashCode() + out.hashCode();
    }

}
