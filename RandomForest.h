#pragma once

#include "DecisionTree.h"

namespace dt {

using RandomForest = vector<unique_ptr<DecisionTree>>;

float evalForest(const RandomForest& forest, const FeatureVector& features);

RandomForest trainForest(ExampleIt first,
                         ExampleIt last,
                         size_t numTrees = 100,
                         float sampleRate = 0.7,
                         size_t maxDepth = 3,
                         float minGain = 0);

float validateForest(const RandomForest& forest,
                     ExampleIt first,
                     ExampleIt last);

}

