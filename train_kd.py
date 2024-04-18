import torch
import torch.nn as nn
import torch.optim as optim


from dataloader import *
from config import *
from model import *

#Called Teacher and student AUtoencoders
teacherAE = Autoencoder().to(device)
studentAE = Autoencoder().to(device)

checkpoint = torch.load(CKPT_teacher)
teacherAE.load_state_dict(checkpoint['teacher_state_dict'])

# teacher and students parameters
teacherOptim = torch.optim.Adam(teacherAE.decoder.parameters(), lr=learning_rate,weight_decay=weight_decay)
studentOptim = torch.optim.Adam(studentAE.encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)


train_loader = training_data_mvtec()

for epoch in range(num_epochs_kd):
    teacher_running_loss = 0.0
    Student_running_loss = 0.0
    for img,_ in train_loader:
        #Teacher Input
        Xi = img.to(device)

        zt = teacherAE.encoder(Xi).detach()
        Xt = teacherAE.decoder(zt)

        # zs is in encoded form
        zs = studentAE.encoder(Xi)

        teacherOptim.zero_grad()
        lossT = mse(Xt, Xi)
        lossT.backward()
        teacherOptim.step()


        lossS = ((lambda_param * mse(zs, zt)) + ((1 - lambda_param) * (1 - cos(zs, zt)))).sum()
        studentOptim.zero_grad()
        lossS.backward()
        studentOptim.step()


        # teacher_running_loss += lossT.item()
        Student_running_loss += lossS.item()
    print(f"Epoch [{epoch + 1}/{num_epochs_kd}], "
          # f"Loss: {teacher_running_loss / len(train_loader):.4f} , "
          f"SLoss: {Student_running_loss / len(train_loader):.4f}"
          )


# save the Training

torch.save({
    'epoch': num_epochs_kd,
    'teacher_model_state_dict': teacherAE.state_dict(), # We will updae it in future
    'student_model_state_dict': studentAE.state_dict(),
    'teacher_optimizer_state_dict': teacherOptim.state_dict(),
    'student_optimizer_state_dict': studentOptim.state_dict(),
    'learning_rate': learning_rate,
    }, CKPT_PATH)