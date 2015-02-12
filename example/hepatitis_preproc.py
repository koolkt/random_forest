examples = []

# Load raw data
with open("hepatitis.data", "r") as inputData:
    for line in inputData:
        attributes = line.split(',')
        label = int(attributes[0] == '2')
        features = attributes[1:]
        examples.append((label, features))

if not examples:
    exit

# Compute mean and range for each feature
numFeatures = len(examples[0][1])
means = []
ranges = []

for i in range(numFeatures):
    total = 0
    minValue = float('Inf')
    maxValue = float('-Inf')
    for e in examples:
        f = e[1][i]
        if f != '?':
            f = float(f)
            total += f
            minValue = min(minValue, f)
            maxValue = max(maxValue, f)
    means.append(total / len(examples))
    ranges.append(maxValue - minValue)

# Produce normalized examples
normalizedExamples = []

for ex in examples:
    newFeatures = []
    for i, f in enumerate(ex[1]):
        if f == '?':
            # Unknown attributes are replaced with mean
            newFeatures.append(0.0)
        else:
            newFeatures.append((float(f) - means[i]) / ranges[i])
    normalizedExamples.append((ex[0], newFeatures))

# Output processed dataset
with open("hepatitis_preproc.data", "w") as outputData:
    for ex in normalizedExamples:
        flattened = [ex[0]] + ex[1]
        outputData.write(' '.join(map(str, flattened)) + '\n')
