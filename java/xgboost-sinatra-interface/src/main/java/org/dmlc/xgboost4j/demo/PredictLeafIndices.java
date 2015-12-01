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
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.dmlc.xgboost4j.Booster;
import org.dmlc.xgboost4j.DMatrix;
import org.dmlc.xgboost4j.demo.util.Params;
import org.dmlc.xgboost4j.util.Trainer;
import org.dmlc.xgboost4j.util.XGBoostError;

/**
 * predict leaf indices
 *
 * @author hzx
 */
public class PredictLeafIndices {
    public Params params = new Params();

    public void start() throws XGBoostError {
        // load file from text file, also binary buffer generated by xgboost4j
        final DMatrix trainMat = new DMatrix("../../demo/data/agaricus.txt.train");
        final DMatrix testMat = new DMatrix("../../demo/data/agaricus.txt.test");

        // specify parameters
        /*final Params param = new Params() {
            {
                put("eta", 1.0);
                put("max_depth", 2);
                put("silent", 1);
                put("objective", "binary:logistic");
            }
        };*/

        // specify watchList
        final List<Map.Entry<String, DMatrix>> watchs = new ArrayList<>();
        watchs.add(new AbstractMap.SimpleEntry<>("train", trainMat));
        watchs.add(new AbstractMap.SimpleEntry<>("test", testMat));

        // train a booster
        final int round = 3;
        final Booster booster = Trainer.train(this.params, trainMat, round, watchs, null, null);

        // predict using first 2 tree
        float[][] leafindex = booster.predict(testMat, 2, true);
        for (final float[] leafs : leafindex) {
            System.out.println(Arrays.toString(leafs));
        }

        // predict all trees
        leafindex = booster.predict(testMat, 0, true);
        for (final float[] leafs : leafindex) {
            System.out.println(Arrays.toString(leafs));
        }
    }

}
