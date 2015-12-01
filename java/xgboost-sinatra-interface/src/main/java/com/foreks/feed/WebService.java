package com.foreks.feed;

import static spark.Spark.get;

import java.io.IOException;

import org.dmlc.xgboost4j.demo.BasicWalkThrough;
import org.dmlc.xgboost4j.demo.BoostFromPrediction;
import org.dmlc.xgboost4j.demo.CrossValidation;
import org.dmlc.xgboost4j.demo.CustomObjective;
import org.dmlc.xgboost4j.demo.ExternalMemory;
import org.dmlc.xgboost4j.demo.GeneralizedLinearModel;
import org.dmlc.xgboost4j.demo.PredictFirstNtree;
import org.dmlc.xgboost4j.demo.PredictLeafIndices;
import org.dmlc.xgboost4j.util.XGBoostError;

import spark.Request;

public class WebService {
    static Double  eta         = 0.0;
    static Integer max_depth   = 0;
    static Integer silent      = 0;
    static String  objective   = "";
    static Double  alpha       = 0.0;
    static String  booster     = "";
    static Integer nthread     = 0;
    static Double  gamma       = 0.0;
    static String  eval_metric = "";

    public static void main(final String[] args) {
        get("/PredictLeafIndices", (request, response) -> predictLeafIndices(request));
        get("/PredictFirstNtree", (request, response) -> predictFirstNtree(request));
        get("/GeneralizedLinearModel", (request, response) -> generalizedLinearModel(request));
        get("/ExternalMemory", (request, response) -> externalMemory(request));
        get("/CustomObjective", (request, response) -> customObjective(request));
        get("/CrossValidation", (request, response) -> crossValidation(request));
        get("/BoostFromPrediction", (request, response) -> boostFromPrediction(request));
        get("/BasicWalkThrough", (request, response) -> basicWalkThrough(request));
    }

    private static Object basicWalkThrough(final Request request) {
        eta = Double.parseDouble(request.headers("eta"));
        max_depth = Integer.parseInt(request.headers("max_depth"));
        silent = Integer.parseInt(request.headers("silent"));
        objective = request.headers("objective");
        final BasicWalkThrough basicWalkThrough = new BasicWalkThrough();
        basicWalkThrough.params.put("eta", eta);
        basicWalkThrough.params.put("max_depth", max_depth);
        basicWalkThrough.params.put("silent", silent);
        basicWalkThrough.params.put("objective", objective);
        try {
            basicWalkThrough.start();
        } catch (IOException | XGBoostError e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return "Everything is OK";
    }

    private static Object boostFromPrediction(final Request request) {
        eta = Double.parseDouble(request.headers("eta"));
        max_depth = Integer.parseInt(request.headers("max_depth"));
        silent = Integer.parseInt(request.headers("silent"));
        objective = request.headers("objective");
        final BoostFromPrediction boostFromPrediction = new BoostFromPrediction();
        boostFromPrediction.params.put("eta", eta);
        boostFromPrediction.params.put("max_depth", max_depth);
        boostFromPrediction.params.put("silent", silent);
        boostFromPrediction.params.put("objective", objective);
        try {
            boostFromPrediction.start();
        } catch (final XGBoostError e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return "Everything is OK";
    }

    private static Object crossValidation(final Request request) {
        eta = Double.parseDouble(request.headers("eta"));
        max_depth = Integer.parseInt(request.headers("max_depth"));
        silent = Integer.parseInt(request.headers("silent"));
        objective = request.headers("objective");
        nthread = Integer.parseInt(request.headers("nthread"));
        gamma = Double.parseDouble(request.headers("gamma"));
        eval_metric = request.headers("eval_metric");
        final CrossValidation crossValidation = new CrossValidation();
        crossValidation.params.put("eta", eta);
        crossValidation.params.put("max_depth", max_depth);
        crossValidation.params.put("silent", silent);
        crossValidation.params.put("nthread", nthread);
        crossValidation.params.put("objectiv", objective);
        crossValidation.params.put("gamma", gamma);
        crossValidation.params.put("eval_metric", eval_metric);
        try {
            crossValidation.start();
        } catch (IOException | XGBoostError e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return "Everything is OK";
    }

    private static Object customObjective(final Request request) {
        eta = Double.parseDouble(request.headers("eta"));
        max_depth = Integer.parseInt(request.headers("max_depth"));
        silent = Integer.parseInt(request.headers("silent"));
        final CustomObjective customObjective = new CustomObjective();
        customObjective.params.put("eta", eta);
        customObjective.params.put("max_depth", max_depth);
        customObjective.params.put("silent", silent);
        try {
            customObjective.start();
        } catch (final XGBoostError e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return "Everything is OK";
    }

    private static Object externalMemory(final Request request) {
        eta = Double.parseDouble(request.headers("eta"));
        max_depth = Integer.parseInt(request.headers("max_depth"));
        silent = Integer.parseInt(request.headers("silent"));
        objective = request.headers("objective");
        final ExternalMemory externalMemory = new ExternalMemory();
        externalMemory.params.put("eta", eta);
        externalMemory.params.put("max_depth", max_depth);
        externalMemory.params.put("silent", silent);
        externalMemory.params.put("objective", objective);
        return "Everything is OK";
    }

    private static Object generalizedLinearModel(final Request request) {
        alpha = Double.parseDouble(request.headers("alpha"));
        silent = Integer.parseInt(request.headers("silent"));
        objective = request.headers("objective");
        booster = request.headers("booster");
        final GeneralizedLinearModel generalizedLinearModel = new GeneralizedLinearModel();
        generalizedLinearModel.params.put("alpha", alpha);
        generalizedLinearModel.params.put("silent", silent);
        generalizedLinearModel.params.put("objective", objective);
        generalizedLinearModel.params.put("booster", booster);
        try {
            generalizedLinearModel.start();
        } catch (final XGBoostError e) {
            e.printStackTrace();
        }
        return "Everything is OK";
    }

    private static Object predictFirstNtree(final Request request) {
        eta = Double.parseDouble(request.headers("eta"));
        max_depth = Integer.parseInt(request.headers("max_depth"));
        silent = Integer.parseInt(request.headers("silent"));
        objective = request.headers("objective");
        final PredictFirstNtree predictFirstNtree = new PredictFirstNtree();
        predictFirstNtree.params.put("eta", eta);
        predictFirstNtree.params.put("max_depth", max_depth);
        predictFirstNtree.params.put("silent", silent);
        predictFirstNtree.params.put("objective", objective);
        try {
            predictFirstNtree.start();
        } catch (final XGBoostError e) {
            e.printStackTrace();
        }
        return "Everything is OK";
    }

    public static Object predictLeafIndices(final Request request) {
        eta = Double.parseDouble(request.headers("eta"));
        max_depth = Integer.parseInt(request.headers("max_depth"));
        silent = Integer.parseInt(request.headers("silent"));
        objective = request.headers("objective");
        final PredictLeafIndices predictLeafIndices = new PredictLeafIndices();
        predictLeafIndices.params.put("eta", eta);
        predictLeafIndices.params.put("max_depth", max_depth);
        predictLeafIndices.params.put("silent", silent);
        predictLeafIndices.params.put("objective", objective);
        try {
            predictLeafIndices.start();
        } catch (final XGBoostError e) {
            e.printStackTrace();
        }
        return "Everything is OK";
    }
}
