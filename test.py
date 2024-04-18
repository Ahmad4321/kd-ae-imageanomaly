import matplotlib.pyplot as plt
from dataloader import *
from config import *
from model import *
from sklearn.metrics import accuracy_score, precision_score, recall_score

#Called Teacher and student Autoencoders
teacherAE = Autoencoder().to(device)
studentAE = Autoencoder().to(device)

checkpoint = torch.load(CKPT_PATH)

# Teacher state
teacherAE.load_state_dict(checkpoint['teacher_model_state_dict'])
#Student State
studentAE.load_state_dict(checkpoint['student_model_state_dict'])

test_loader = testing_data_mvtec()

anomaly_scores = []

true_labels = []
for data,label in test_loader:
    Xi = data.to(device)
    zt = teacherAE.encoder(Xi)
    zs = studentAE.encoder(Xi)

    score = (0.3 * mse(zs, zt) + 0.3 * (1 - cos(zs, zt)) + 0.4 * mse(teacherAE.decoder(zs), Xi))
    scoremean = torch.mean(score)
     # scoremean.item()  > 0.5
    anomaly_scores.append(scoremean.item())
    true_labels.append(int(label))

print(anomaly_scores)

# Perform anomaly detection evaluation
predicted_labels = [1 if score > anomaly_threshold else 0 for score in anomaly_scores]
print(true_labels)
print(predicted_labels)

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels,zero_division=1)
recall = recall_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

plt.plot(anomaly_scores, marker='o')
plt.title('Anomaly Scores')
plt.xlabel('Image Index')
plt.ylabel('Anomaly Score')
plt.grid(True)
plt.show()

plt.savefig('./save/foo1.png')
