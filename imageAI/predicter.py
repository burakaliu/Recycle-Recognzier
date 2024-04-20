from imageai.Classification import ImageClassification
import os

execution_path = os.getcwd()

prediction = ImageClassification()
prediction.setModelTypeAsDenseNet121()
prediction.setModelPath(os.path.join(execution_path, r"C:\Users\burak\Documents\Hackathon\Recycle-Recognzier\imageAI\densenet121-a639ec97.pth"))
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, r"C:\Users\burak\Documents\Hackathon\Recycle-Recognzier\imageAI\pizzabed.png"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)