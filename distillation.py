import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

from utils import set_seed


def train_kd(
    rnd,
    epoch,
    teacher_model,
    student_model,
    data_raw,
    data_grid,
    lr=0.00001,
    temperature=1.0,
    alpha=0.5,
    mode="offline",
    seed=None,
):
    # reproducibility
    if seed is not None:
        set_seed(seed)

    # move models to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model.to(device)
    teacher_model.to(device)
    
    # loss functions
    criterion_hard = nn.CrossEntropyLoss().to(device)
    criterion_soft = nn.KLDivLoss(reduction="batchmean").to(device)

    # optimizers
    optimizer_student = optim.Adam(student_model.parameters(), lr=lr)
    optimizer_teacher = (
        optim.Adam(teacher_model.parameters(), lr=lr) if mode == "online" else None
    )

    # set models to train mode
    student_model.train()
    teacher_model.train() if mode == "online" else teacher_model.eval()

    # initialize counters
    running_loss_student = 0.0
    running_loss_teacher = 0.0
    correct_student = 0
    correct_teacher = 0
    total = 0

    # training loop
    progress_bar = tqdm(
        enumerate(zip(data_raw, data_grid)),
        total=len(data_raw),
        desc=f"Training Round {rnd} | Epoch {epoch+1}",
        leave=False,
    )
    for i, (raw, grid) in progress_bar:
        # unpacking raw and grid data
        X_raw, y_raw = raw
        X_grid, y_grid = grid

        X_raw, y_raw = X_raw.to(device), y_raw.to(device)
        X_grid, y_grid = X_grid.to(device), y_grid.to(device)

        assert torch.equal(y_raw, y_grid), "Both y must be equal"
        y = y_raw

        # zero gradients
        optimizer_student.zero_grad()
        if optimizer_teacher:
            optimizer_teacher.zero_grad()

        # forward pass for both models
        teacher_outputs = teacher_model(X_raw)  # Y psi
        student_outputs = student_model(X_grid)  # Y phi

        # calculate the hard loss (L_CE)
        loss_teacher_hard = criterion_hard(teacher_outputs, y)  # L_CE psi
        loss_student_hard = criterion_hard(student_outputs, y)  # L_CE phi

        # calculate the soft loss (L_KL)
        teacher_softmax = F.softmax(teacher_outputs / temperature, dim=1)  # Q
        student_log_softmax = F.log_softmax(student_outputs / temperature, dim=1)  # P
        loss_soft_student = criterion_soft(
            student_log_softmax, teacher_softmax.detach()
        )  # student uses teacher's output
        loss_soft_teacher = criterion_soft(
            student_log_softmax.detach(), teacher_softmax
        )  # teacher uses student's output

        # calculate the combined loss
        loss_teacher = (
            alpha * loss_soft_teacher + (1 - alpha) * loss_teacher_hard
        )  # L_TOTAL psi
        loss_student = (
            alpha * loss_soft_student + (1 - alpha) * loss_student_hard
        )  # L_TOTAL phi

        # backward pass and optimizer step for student model
        loss_student.backward(retain_graph=True)
        optimizer_student.step()

        # backward pass and optimizer step for teacher model
        if mode == "online":
            loss_teacher.backward()
            optimizer_teacher.step()

        # update counters
        running_loss_student += loss_student.item()
        running_loss_teacher += loss_teacher.item()
        _, predicted_student = torch.max(student_outputs.data, 1)
        _, predicted_teacher = torch.max(teacher_outputs.data, 1)
        total += y.size(0)
        correct_student += (predicted_student == y).sum().item()
        correct_teacher += (predicted_teacher == y).sum().item()

        # update progress bar
        current_loss_student = running_loss_student / (i + 1)
        current_loss_teacher = running_loss_teacher / (i + 1)
        accuracy_student = 100 * correct_student / total
        accuracy_teacher = 100 * correct_teacher / total
        progress_bar.set_postfix(
            Student_Loss=f"{current_loss_student:.4f}",
            Student_Accuracy=f"{accuracy_student:.2f}%",
            Teacher_Loss=f"{current_loss_teacher:.4f}",
            Teacher_Accuracy=f"{accuracy_teacher:.2f}%",
        )

    # calculate average loss and accuracy
    average_loss_student = running_loss_student / len(data_raw)
    average_loss_teacher = running_loss_teacher / len(data_raw)
    accuracy_student = 100 * correct_student / total
    accuracy_teacher = 100 * correct_teacher / total
    
    # return the metrics
    return (
        average_loss_student,
        average_loss_teacher,
        accuracy_student,
        accuracy_teacher,
    )


def validate_kd(
    rnd,
    epoch,
    teacher_model,
    student_model,
    data_raw,
    data_grid,
    temperature=1.0,
    alpha=0.5,
    mode="offline",
    seed=None,
):
    # reproducibility
    if seed is not None:
        set_seed(seed)

    # move models to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model.to(device)
    teacher_model.to(device)
    
    # loss functions
    criterion_hard = nn.CrossEntropyLoss().to(device)
    criterion_soft = nn.KLDivLoss(reduction="batchmean").to(device)

    # set models to evaluation mode
    student_model.eval()
    teacher_model.eval()

    # initialize counters
    running_loss_student = 0.0
    running_loss_teacher = 0.0
    correct_student = 0
    correct_teacher = 0
    total = 0

    # validation loop
    progress_bar = tqdm(
        enumerate(zip(data_raw, data_grid)),
        total=len(data_raw),
        desc=f"Validation Round {rnd} | Epoch {epoch+1}",
        leave=False,
    )
    with torch.no_grad():
        for i, (raw, grid) in progress_bar:
            # unpacking raw and grid data
            X_raw, y_raw = raw
            X_grid, y_grid = grid

            X_raw, y_raw = X_raw.to(device), y_raw.to(device)
            X_grid, y_grid = X_grid.to(device), y_grid.to(device)

            assert torch.equal(y_raw, y_grid), "Both y must be equal"
            y = y_raw

            # forward pass for both models
            teacher_outputs = teacher_model(X_raw)
            student_outputs = student_model(X_grid)

            # loss calculation
            loss_teacher_hard = criterion_hard(teacher_outputs, y)
            loss_student_hard = criterion_hard(student_outputs, y)

            loss_soft = criterion_soft(
                F.log_softmax(student_outputs / temperature, dim=1),
                F.softmax(teacher_outputs / temperature, dim=1),
            )

            # calculate combined losses
            loss_student = alpha * loss_soft + (1 - alpha) * loss_student_hard
            running_loss_student += loss_student.item()

            loss_teacher = alpha * loss_soft.detach() + (1 - alpha) * loss_teacher_hard
            running_loss_teacher += loss_teacher.item()

            # update counters
            _, predicted_student = torch.max(student_outputs.data, 1)
            _, predicted_teacher = torch.max(teacher_outputs.data, 1)
            total += y.size(0)
            correct_student += (predicted_student == y).sum().item()
            correct_teacher += (predicted_teacher == y).sum().item()

            # update progress bar
            accuracy_student = 100 * correct_student / total
            accuracy_teacher = 100 * correct_teacher / total
            current_loss_student = running_loss_student / (i + 1)
            current_loss_teacher = running_loss_teacher / (i + 1)

            progress_bar.set_postfix(
                Student_Loss=f"{current_loss_student:.4f}",
                Student_Accuracy=f"{accuracy_student:.2f}%",
                Teacher_Loss=f"{current_loss_teacher:.4f}",
                Teacher_Accuracy=f"{accuracy_teacher:.2f}%",
            )

    # calculate average loss and accuracy
    average_loss_student = running_loss_student / len(data_raw)
    average_loss_teacher = running_loss_teacher / len(data_raw)
    accuracy_student = 100 * correct_student / total
    accuracy_teacher = 100 * correct_teacher / total
    
    # return the metrics
    return (
        average_loss_student,
        average_loss_teacher,
        accuracy_student,
        accuracy_teacher,
    )


def test_model(model, dataloader):
    
    # move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # loss function
    criterion = nn.CrossEntropyLoss().to(device)
    
    # set model to evaluation mode
    model.eval()

    # initialize counters
    running_loss = 0.0
    correct = 0
    total = 0

    # testing loop
    progress_bar = tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Testing", leave=False
    )
    with torch.no_grad():
        for i, (X, y) in progress_bar:
            # move data to device
            X, y = X.to(device), y.to(device)

            # forward pass
            outputs = model(X)
            
            # calculate loss
            loss = criterion(outputs, y)

            # update counters
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            # update progress bar
            current_loss = running_loss / (i + 1)
            accuracy = 100 * correct / total
            progress_bar.set_postfix(
                Loss=f"{current_loss:.4f}", Accuracy=f"{accuracy:.2f}%"
            )

    # calculate average loss and accuracy
    average_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    # return the metrics
    return average_loss, accuracy
