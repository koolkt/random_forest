#pragma once

#include <memory>
#include <vector>

namespace dt {

using std::unique_ptr;
using std::vector;

using FeatureVector = vector<float>;

struct Example {
  FeatureVector features;
  bool label;
};

using ExampleIt = vector<Example>::iterator;

struct DecisionTree {
  size_t splitFeature;
  float splitValue;

  // Probability of positive at this node
  float value;
  
  // Split feature < or >= split value
  unique_ptr<DecisionTree> left, right;
};

float evalTree(DecisionTree* tree, const FeatureVector& features);

unique_ptr<DecisionTree> trainTree(ExampleIt first,
                                   ExampleIt last,
                                   size_t maxDepth = 3,
                                   float minGain = 0);

// Returns accuracy
float validateTree(DecisionTree* tree, ExampleIt first, ExampleIt last);

}
