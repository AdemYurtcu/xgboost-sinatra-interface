/*
 * Copyright (c) 2014 by Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a
 * copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations
 * under the License.
 */
package org.dmlc.xgboost4j.demo;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.dmlc.xgboost4j.Booster;
import org.dmlc.xgboost4j.DMatrix;
import org.dmlc.xgboost4j.IEvaluation;
import org.dmlc.xgboost4j.IObjective;
import org.dmlc.xgboost4j.demo.util.Params;
import org.dmlc.xgboost4j.util.Trainer;
import org.dmlc.xgboost4j.util.XGBoostError;

/**
 * an example user define objective and eval NOTE: when you do customized loss function, the default prediction value is margin this may make buildin
 * evalution metric not function properly for example, we are doing logistic loss, the prediction is score before logistic transformation he buildin
 * evaluation error assumes input is after logistic transformation Take this in mind when you use the customization, and maybe you need write
 * customized evaluation function
 *
 * @author hzx
 */
public class CustomObjective {
    /**
     * loglikelihoode loss obj function
     */
    public static class LogRegObj implements IObjective {
        private static final Log logger = LogFactory.getLog(LogRegObj.class);

        /**
         * simple sigmoid func
         *
         * @param input
         * @return Note: this func is not concern about numerical stability, only used as example
         */
        public float sigmoid(final float input) {
            final float val = (float) (1 / (1 + Math.exp(-input)));
            return val;
        }

        public float[][] transform(final float[][] predicts) {
            final int nrow = predicts.length;
            final float[][] transPredicts = new float[nrow][1];

            for (int i = 0; i < nrow; i++) {
                transPredicts[i][0] = sigmoid(predicts[i][0]);
            }

            return transPredicts;
        }

        @Override
        public List<float[]> getGradient(final float[][] predicts, final DMatrix dtrain) {
            final int nrow = predicts.length;
            final List<float[]> gradients = new ArrayList<>();
            float[] labels;
            try {
                labels = dtrain.getLabel();
            } catch (final XGBoostError ex) {
                logger.error(ex);
                return null;
            }
            final float[] grad = new float[nrow];
            final float[] hess = new float[nrow];

            final float[][] transPredicts = transform(predicts);

            for (int i = 0; i < nrow; i++) {
                final float predict = transPredicts[i][0];
                grad[i] = predict - labels[i];
                hess[i] = predict * (1 - predict);
            }

            gradients.add(grad);
            gradients.add(hess);
            return gradients;
        }
    }

    /**
     * user defined eval function. NOTE: when you do customized loss function, the default prediction value is margin this may make buildin evalution
     * metric not function properly for example, we are doing logistic loss, the prediction is score before logistic transformation the buildin
     * evaluation error assumes input is after logistic transformation Take this in mind when you use the customization, and maybe you need write
     * customized evaluation function
     */
    public static class EvalError implements IEvaluation {
        private static final Log logger = LogFactory.getLog(EvalError.class);

        String evalMetric = "custom_error";

        public EvalError() {
        }

        @Override
        public String getMetric() {
            return this.evalMetric;
        }

        @Override
        public float eval(final float[][] predicts, final DMatrix dmat) {
            float error = 0f;
            float[] labels;
            try {
                labels = dmat.getLabel();
            } catch (final XGBoostError ex) {
                logger.error(ex);
                return -1f;
            }
            final int nrow = predicts.length;
            for (int i = 0; i < nrow; i++) {
                if (labels[i] == 0f && predicts[i][0] > 0) {
                    error++;
                } else if (labels[i] == 1f && predicts[i][0] <= 0) {
                    error++;
                }
            }

            return error / labels.length;
        }
    }
    public Params params = new Params();

    public void start() throws XGBoostError {
        // load train mat (svmlight format)
        final DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
        // load valid mat (svmlight format)
        final DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test");

        // set params
        // set params
        /*final Params param = new Params() {
            {
                put("eta", 1.0);
                put("max_depth", 2);
                put("silent", 1);
            }
        };*/

        // set round
        final int round = 2;

        // specify watchList
        final List<Map.Entry<String, DMatrix>> watchs = new ArrayList<>();
        watchs.add(new AbstractMap.SimpleEntry<>("train", trainMat));
        watchs.add(new AbstractMap.SimpleEntry<>("test", testMat));

        // user define obj and eval
        final IObjective obj = new LogRegObj();
        final IEvaluation eval = new EvalError();

        // train a booster
        System.out.println("begin to train the booster model");
        final Booster booster = Trainer.train(this.params, trainMat, round, watchs, obj, eval);
    }
}
