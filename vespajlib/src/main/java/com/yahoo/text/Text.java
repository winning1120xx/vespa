// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.text;

import java.util.Locale;
import java.util.OptionalInt;

/**
 * Text utility functions. See also {com.yahoo.language.process.CharacterClasses}.
 * 
 * @author bratseth
 */
public final class Text {

    private static final boolean[] allowedAsciiChars = new boolean[0x80];

    static {
        allowedAsciiChars[0x0] = false;
        allowedAsciiChars[0x1] = false;
        allowedAsciiChars[0x2] = false;
        allowedAsciiChars[0x3] = false;
        allowedAsciiChars[0x4] = false;
        allowedAsciiChars[0x5] = false;
        allowedAsciiChars[0x6] = false;
        allowedAsciiChars[0x7] = false;
        allowedAsciiChars[0x8] = false;
        allowedAsciiChars[0x9] = true;  //tab
        allowedAsciiChars[0xA] = true;  //nl
        allowedAsciiChars[0xB] = false;
        allowedAsciiChars[0xC] = false;
        allowedAsciiChars[0xD] = true;  //cr
        for (int i = 0xE; i < 0x20; i++) {
            allowedAsciiChars[i] = false;
        }
        for (int i = 0x20; i < 0x7F; i++) {
            allowedAsciiChars[i] = true;  //printable ascii chars
        }
        allowedAsciiChars[0x7F] = true;  //del - discouraged, but allowed
    }

    /** No instantiation */
    private Text() {}

    /**
     * Returns whether the given codepoint is a valid text character, potentially suitable for 
     * purposes such as indexing and display, see http://www.w3.org/TR/2006/REC-xml11-20060816/#charsets
     */
    public static boolean isTextCharacter(int codepoint) {
        // The link above notes that 0x7F-0x84 and 0x86-0x9F are discouraged, but they are still allowed -
        // see http://www.w3.org/International/questions/qa-controls

        return (codepoint < 0x80)
                ? allowedAsciiChars[codepoint]
                : (codepoint < Character.MIN_SURROGATE) || isTextCharAboveMinSurrogate(codepoint);
    }
    private static boolean isTextCharAboveMinSurrogate(int codepoint) {
        if (codepoint <= Character.MAX_SURROGATE) return false;
        if (codepoint <  0xFDD0)   return true;
        if (codepoint <= 0xFDDF)   return false;
        if (codepoint >= 0x10FFFE) return false;
        return (codepoint & 0xffff) < 0xFFFE;
    }

    /**
     * Validates that the given string value only contains text characters and
     * returns the first illegal code point if one is found.
     */
    public static OptionalInt validateTextString(String string) {
        for (int i = 0; i < string.length(); ) {
            int codePoint = string.codePointAt(i);
            if ( ! Text.isTextCharacter(codePoint))
                return OptionalInt.of(codePoint);

            int charCount = Character.charCount(codePoint);
            if (Character.isHighSurrogate(string.charAt(i))) {
                if ( charCount == 1) {
                    return OptionalInt.of(string.codePointAt(i));
                } else if ( ! Character.isLowSurrogate(string.charAt(i+1))) {
                    return OptionalInt.of(string.codePointAt(i+1));
                }
            }
            i += charCount;
        }
        return OptionalInt.empty();
    }

    /**
     * Validates that the given string value only contains text characters.
     */
    public static boolean isValidTextString(String string) {
        int length = string.length();
        for (int i = 0; i < length; ) {
            int codePoint = string.codePointAt(i);
            if (codePoint < 0x80) {
                if ( ! allowedAsciiChars[codePoint]) return false;
            } else if (codePoint >= Character.MIN_SURROGATE) {
                if ( ! isTextCharAboveMinSurrogate(codePoint)) return false;
                if ( ! Character.isBmpCodePoint(codePoint)) {
                    i++;
                }
            }
            i++;
        }
        return true;
    }


    /** Returns whether the given code point is displayable. */
    public static boolean isDisplayable(int codePoint) {
        return switch (Character.getType(codePoint)) {
            case Character.CONNECTOR_PUNCTUATION, Character.DASH_PUNCTUATION, Character.START_PUNCTUATION,
                 Character.END_PUNCTUATION, Character.INITIAL_QUOTE_PUNCTUATION, Character.FINAL_QUOTE_PUNCTUATION,
                 Character.OTHER_PUNCTUATION, Character.LETTER_NUMBER, Character.OTHER_LETTER,
                 Character.LOWERCASE_LETTER, Character.TITLECASE_LETTER, Character.MODIFIER_LETTER,
                 Character.UPPERCASE_LETTER, Character.DECIMAL_DIGIT_NUMBER, Character.OTHER_NUMBER,
                 Character.CURRENCY_SYMBOL, Character.OTHER_SYMBOL, Character.MATH_SYMBOL -> true;
            default -> false;
        };
    }

    private static StringBuilder lazy(StringBuilder sb, String s, int i) {
        if (sb == null) {
            sb = new StringBuilder(s.substring(0, i));
        }
        sb.append(' ');
        return sb;
    }

    /** Returns a string where any invalid characters in the input string is replaced by spaces */
    public static String stripInvalidCharacters(String string) {
        StringBuilder stripped = null; // lazy, as most string will not need stripping
        for (int i = 0; i < string.length();) {
            int codePoint = string.codePointAt(i);
            int charCount = Character.charCount(codePoint);
            if ( ! Text.isTextCharacter(codePoint)) {
                stripped = lazy(stripped, string, i);
            } else {
                if (Character.isHighSurrogate(string.charAt(i))) {
                    if (charCount == 1) {
                        stripped = lazy(stripped, string, i);
                    } else if (!Character.isLowSurrogate(string.charAt(i+1))) {
                        stripped = lazy(stripped, string, i);
                    } else {
                        if (stripped != null) {
                            stripped.appendCodePoint(codePoint);
                        }
                    }
                } else {
                    if (stripped != null) {
                        stripped.appendCodePoint(codePoint);
                    }
                }
            }
            i += charCount;
        }
        return stripped != null ? stripped.toString() : string;
    }

    /**
     * Returns a string which is never larger than the given number of code points.
     * If the string is longer than the given length it will be truncated.
     * If length is 4 or less the string will be truncated to length.
     * If length is longer than 4, it will be truncated at length-4 with " ..." added at the end.
     */
    public static String truncate(String s, int length) {
        if (s.codePointCount(0, s.length()) <= length) return s;
        if (length <= 4) return substringByCodepoints(s, 0, length);
        return substringByCodepoints(s, 0, length - 4) + " ...";
    }

    public static String substringByCodepoints(String s, int fromCP, int toCP) {
        int len = s.length();
        if ((fromCP >= len) || (fromCP >= toCP)) return "";

        int from;
        try {
            from = s.offsetByCodePoints(0, fromCP);
        } catch (IndexOutOfBoundsException e) {
            return "";
        }
        if (from >= len) return "";
        int lenCP = toCP - fromCP;
        if (from + lenCP >= len) return s.substring(from);

        try {
            int to = s.offsetByCodePoints(from, toCP - fromCP);
            return (to >= len)
                    ? s.substring(from)
                    : s.substring(from, to);
        } catch (IndexOutOfBoundsException e) {
            return s.substring(from);
        }
    }

    public static String format(String format, Object... args) {
	return String.format(Locale.US, format, args);
    }

}
