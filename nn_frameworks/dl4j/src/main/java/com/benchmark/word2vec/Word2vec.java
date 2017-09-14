package com.benchmark.word2vec;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;

import org.apache.commons.cli.*;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.*;
/**
 * Created by agibsonccc on 10/9/14.
 *
 * Neural net that processes text into wordvectors. See below url for an in-depth explanation.
 * https://deeplearning4j.org/word2vec.html
 */
public class Word2vec {

    private static Logger log = LoggerFactory.getLogger(Word2vec.class);

    public static void main(String[] args) throws Exception {
        //Parse command line arguments
        Options options = new Options();

        Option input = new Option("i", "input", true, "Absolute input file path");
        input.setRequired(true);
        options.addOption(input);

        Option output = new Option("o", "output", true, "output file");
        output.setRequired(true);
        options.addOption(output);

        Option epochs = new Option("e", "epochs", true, "Number of epochs to train. Each epoch processes the training data once completely.");
        epochs.setRequired(true);
        options.addOption(epochs);

        Option embedding_size = new Option("s", "embedding_size", true, "The embedding dimension size.");
        embedding_size.setRequired(true);
        options.addOption(embedding_size);

        Option learning_rate = new Option("l", "learning_rate", true, "Initial learning rate.");
        learning_rate.setRequired(true);
        options.addOption(learning_rate);

        Option neg = new Option("n", "neg", true, "Negative samples per training example.");
        neg.setRequired(true);
        options.addOption(neg);

        Option batch_size = new Option("b", "batch_size", true, "Number of training examples processed per step");
        batch_size.setRequired(true);
        options.addOption(batch_size);

        Option window_size = new Option("w", "window_size", true, "The number of words to predict to the left and right");
        window_size.setRequired(true);
        options.addOption(window_size);

        Option min_count = new Option("m", "min_count", true, "The minimum number of word occurrences for it to be included in the vocabulary");
        min_count.setRequired(true);
        options.addOption(min_count);

        Option subsample = new Option("ss", "subsample", true, "Subsample threshold for word occurrence. Words that appear with higher frequency will be randomly down-sampled.");
        subsample.setRequired(true);
        options.addOption(subsample);

        Option workers = new Option("th", "workers", true, "Use these many worker threads to train the model");
        workers.setRequired(true);
        options.addOption(workers);

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd;

        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("utility-name", options);

            System.exit(1);
            return;
        }

        String inputFilePath = cmd.getOptionValue("input");
        String outputFilePath = cmd.getOptionValue("output");
        int epoch_ = Integer.parseInt(cmd.getOptionValue("epochs"));
        int embedding_size_ = Integer.parseInt(cmd.getOptionValue("embedding_size"));
        int neg_ = Integer.parseInt(cmd.getOptionValue("neg"));
        int batch_size_ = Integer.parseInt(cmd.getOptionValue("batch_size"));
        int window_size_ = Integer.parseInt(cmd.getOptionValue("window_size"));
        int min_count_ = Integer.parseInt(cmd.getOptionValue("min_count"));
        int workers_ = Integer.parseInt(cmd.getOptionValue("workers"));
        double subsample_ = Double.parseDouble(cmd.getOptionValue("subsample"));
        double learning_rate_ = Double.parseDouble(cmd.getOptionValue("learning_rate"));

        log.info("Load & Vectorize Sentences....");
        // Strip white space before and after for each line
        SentenceIterator iter = new BasicLineIterator(inputFilePath);
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(min_count_)
                .batchSize(batch_size_)
                .epochs(epoch_)
                .layerSize(embedding_size_)
                .sampling(subsample_)
                .seed(42)
                .useHierarchicSoftmax(false)
                .negativeSample(neg_)
                .learningRate(learning_rate_)
                .windowSize(window_size_)
                .iterate(iter)
                .tokenizerFactory(t)
                .workers(workers_)
                .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();

        log.info("Writing word vectors to text file....");
        System.out.println(vec.vocab(). numWords());

        try (OutputStream outVectorPath = new FileOutputStream(outputFilePath, true)) {

            String firstLine = Integer.toString(vec.vocab(). numWords()) + " " + Integer.toString(embedding_size_) + "\n";
            byte[] bytes = firstLine.getBytes();
            // write a byte sequence
            outVectorPath.write(bytes);

            WordVectorSerializer.writeWordVectors(vec, outVectorPath);

        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
