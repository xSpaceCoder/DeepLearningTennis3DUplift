import torch

# Replace with your actual path
model = torch.hub.load('c:/Users/Alexa/Programmieren/Tennis3DUplifting/DeepLearningTennis3DUplift', 'tennis_uplifting', input_type='rallies', mode='dynamicAnkle', source='local')
print(model)