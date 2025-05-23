// Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.language.opennlp;

import com.yahoo.language.Language;
import com.yahoo.language.detect.Detection;
import com.yahoo.language.detect.Detector;
import com.yahoo.language.detect.Hint;
import com.yahoo.language.simple.SimpleDetector;
import opennlp.tools.langdetect.LanguageDetectorConfig;
import opennlp.tools.langdetect.LanguageDetectorME;
import opennlp.tools.langdetect.LanguageDetectorModel;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

import static java.nio.charset.StandardCharsets.UTF_8;

/**
 * Detects text language using patched OpenNLP, with fallback to {@link SimpleDetector} for undetected CJK input.
 *
 * @author jonmv
 */
class OpenNlpDetector implements Detector {

    private static final Object monitor = new Object();
    private static LanguageDetectorModel model;

    private final SimpleDetector simple = new SimpleDetector();
    private final Map<String, Language> languagesByISO3 = new HashMap<>();
    private final LanguageDetectorME detector;
    private final LanguageDetectorConfig config;
    private final double confidenceThreshold;

    OpenNlpDetector() { this(2.0); }

    OpenNlpDetector(double threshold) {
        detector = new LanguageDetectorME(loadModel());
        config = new LanguageDetectorConfig();
        confidenceThreshold = threshold;
        config.setMinDiff(0.02);
        config.setChunkSize(32);
        config.setMaxLength(256);
        for (Locale locale : Locale.getAvailableLocales()) {
            Language language = Language.fromLocale(locale);
            if (language != null) {
                languagesByISO3.put(locale.getISO3Language(), language);
            }
        }
        var toAdd = new java.util.Properties();
        try {
            var mapStream = OpenNlpDetector.class.getResourceAsStream("/iana/language-subtags.txt");
            toAdd.load(mapStream);
        } catch (java.io.IOException|NullPointerException e) {
            throw new IllegalStateException("Could not load subtags from resource");
        }
        for (String subtag : toAdd.stringPropertyNames()) {
            String tag = toAdd.getProperty(subtag);
            if (! languagesByISO3.containsKey(subtag)) {
                var locale = Locale.forLanguageTag(tag);
                if (locale != null) {
                    var lang = Language.fromLocale(locale);
                    if (lang != null && lang != Language.UNKNOWN) {
                        languagesByISO3.put(subtag, lang);
                    }
                }
            }
        }
    }

    private static LanguageDetectorModel loadModel() {
        synchronized (monitor) {
            if (model == null) {
                try {
                    model = new LanguageDetectorModel(OpenNlpDetector.class.getResourceAsStream("/models/langdetect-183.bin"));
                }
                catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
        }
        return model;
    }

    @Override
    public Detection detect(byte[] input, int offset, int length, Hint hint) {
        Charset encoding = Charset.forName(simple.guessEncoding(input, offset, length));
        return new Detection(detectLanguage(new String(input, offset, length, encoding)), encoding.name(), false);
    }

    @Override
    public Detection detect(ByteBuffer input, Hint hint) {
        if (input.hasArray())
            return detect(input.array(), input.arrayOffset() + input.position(), input.remaining(), hint);

        byte[] buffer = new byte[input.remaining()];
        input.get(buffer);
        return detect(buffer, 0, buffer.length, hint);
    }

    @Override
    public Detection detect(String input, Hint hint) {
        return new Detection(detectLanguage(input), UTF_8.name(), false);
    }

    private Language detectLanguage(String input) {
        var prediction = detector.probingPredictLanguages(input, config).languages()[0];
        // getConfidence() is typically from 0.01 (very low) to 0.10 (very high).
        // let the configured threshold have reasonable values from 1 to 10:
        double confidence = 100.0 * prediction.getConfidence();
        Language result = null;
        if (confidence > confidenceThreshold) {
            result = languagesByISO3.get(prediction.getLang());
        }
        if (result == null) {
            result = simple.guessLanguage(input.substring(0, Math.min(input.length(), 256)));
        }
        return result;
    }

}
