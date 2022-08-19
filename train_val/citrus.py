import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import time
import os
import sys
from customdataset_3 import CustomImageDataset
from confusion_matrix_210825 import confusion_matrix
from efficientnet_pytorch import EfficientNet
from pytorch_pretrained_vit import ViT


# train_loader : train data 불러오기
# val_loader : validation data 불러오기
# device : cuda
# optimizer : optimizer (Adam)
# net : model
# criterion : CrossEntropyLoss
# scheduler :CosineAnnealingLR
# train_set : train data
# val_set : validation data
# num_classes : class 갯수
# model_name : Resnet50, Densenet161
# actual_class : confusion matrix 배열
# ex_path : log 저장 path

def run(train_loader, val_loader, device, optimizer, net, criterion, scheduler, train_set, val_set, num_classes,
        model_name, actual_class, ex_path):
    setSysoutpath(ex_path)  # 표준 출력을 파일 출력으로 변경
    now = time.localtime()  # 현재 시간

    acc_max = 0  # best accuracy
    f1_max = 0  # best f1 score
    epoch = 3  # epoch

    epochs = []  # figure epoch
    train_acc = []
    validation_acc = []
    f1_list = []

    for epoch in range(epoch):
        epochs.append(epoch)
        epoch_start = time.localtime()

        # Training timestamp
        print("Start train.py")
        print("[%04d/%02d/%02d %02d:%02d:%02d] 감귤 %s Train_start" % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec, model_name))

        class_names = train_set.class_names
        print(f"class names: {class_names}")

        print("[%04d/%02d/%02d %02d:%02d:%02d] %dth epoch_started" % (
            epoch_start.tm_year, epoch_start.tm_mon, epoch_start.tm_mday, epoch_start.tm_hour, epoch_start.tm_min,
            epoch_start.tm_sec, epoch + 1))

        # Training 시작
        e_loss, e_acc = doTrain(train_loader, net, device, optimizer, criterion, scheduler, epoch, train_set)
        print('train loss ', e_loss, ' epoch: ', epoch + 1)
        print('train acc ', e_acc, ' epoch: ', epoch + 1)

        train_acc.append(e_acc)

        end = time.localtime()
        print("[%04d/%02d/%02d %02d:%02d:%02d] Training_finished" % (
            end.tm_year, end.tm_mon, end.tm_mday, end.tm_hour, end.tm_min, end.tm_sec))
        print("")

        # Validation timestamp
        print("Start validation.py")
        print("[%04d/%02d/%02d %02d:%02d:%02d] 감귤 %s Validation_start" % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec, model_name))

        class_names = val_set.class_names
        print(f"class names: {class_names}")

        print("[%04d/%02d/%02d %02d:%02d:%02d] %dth epoch_validation" % (
            epoch_start.tm_year, epoch_start.tm_mon, epoch_start.tm_mday, epoch_start.tm_hour, epoch_start.tm_min,
            epoch_start.tm_sec, epoch + 1))

        # Validation 시작
        e_acc, f1_score = doValidation(val_loader, device, net, actual_class, model_name, val_set, num_classes)

        validation_acc.append(e_acc)

        # best f1 score, validation accuracy
        f1_max = max(f1_max, f1_score)

        if True:
            save_checkpoint(net, f'./{ex_path}/best_{model_name}-{epoch}.pth')

        acc_max = max(acc_max, e_acc)

        # figure : f1 score vertical axis
        f1_list.append(f1_score)

    print('Finished Validation')
    print(f'Best {model_name} Acc :', acc_max)
    print(f'Best F1 score :', f1_max)

    figure(epochs, train_acc, f'./{ex_path}/result_train_{ex_path}.jpg')
    figure(epochs, validation_acc, f'./{ex_path}/result_valid_{ex_path}.jpg')
    figure(epochs, f1_list, f'./{ex_path}/result_f1_{ex_path}.jpg')

    sys.stdout.flush()


def figure(x_list, y_list, save_path):
    plt.clf()
    plt.plot(x_list, y_list)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# Training
def doTrain(train_loader, net, device, optimizer, criterion, scheduler, epoch, train_set):
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    total_loss = 0
    total_correct = 0
    net.train()  # Convert the model to training mode

    for i, data in loop:
        inputs = data['image'].to(device)
        labels = data['label'].to(device)

        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)

        train_loss = loss.item()
        train_correct = torch.sum(predicted == labels.data).item()

        total_loss += train_loss
        total_correct += train_correct

    now = time.localtime()
    current_time = "%04d/%02d/%02d %02d:%02d:%02d" % (
        now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    loop.set_description(f"[{current_time}] Epoch [{epoch + 1}/{epoch}]")

    loop.set_postfix(loss=train_loss, acc=train_correct / len(labels))

    scheduler.step()

    # loss 값 계산
    e_loss = total_loss / len(train_set)

    # 정확도 계산
    e_acc = total_correct / len(train_set)

    return e_loss, e_acc


# Validation
def doValidation(val_loader, device, net, actual_class, model_name, val_set, num_classes):
    net.eval()  # Convert the model to validation mode

    class_names = val_set.class_names

    # 모델 데이터 검증
    total_correct, actual_data, predicted, image_path_list = process(val_loader,
                                                                     device,
                                                                     net,
                                                                     actual_class)

    # 정확도 계산
    e_acc = total_correct / len(val_set)

    # GT 값 계산
    getGroundTruth(actual_data, predicted, image_path_list)

    # F1 Score 계산
    f1_score = getF1Score(predicted, actual_data, model_name, actual_class, num_classes, class_names)

    end = time.localtime()
    print("[%04d/%02d/%02d %02d:%02d:%02d] Validation_finished" % (
        end.tm_year, end.tm_mon, end.tm_mday, end.tm_hour, end.tm_min, end.tm_sec))
    print("-----------------------------------------------------------------------------------------------------")
    print(" ")

    return e_acc, f1_score


# 모델 데이터 검증
def process(val_loader, device, net, actual_class):
    with torch.no_grad():
        total_correct = 0  # 정확도
        loop = tqdm(enumerate(val_loader), total=len(val_loader))
        image_path_list = []  # 이미지 path list
        actual_data = []  # 정답 list
        predict_data = []  # 예측 list

        for i, data in loop:
            inputs = data['image'].to(device)
            labels = data['label'].to(device)
            image_path = data['image_path']

            outputs = net(inputs)

            _, predicted = torch.max(outputs, 1)

            val_correct = torch.sum(predicted == labels.data).item()

            total_correct += val_correct

            labels_list = labels.tolist()
            predicted_label_list = predicted.tolist()

            actual_class[labels_list[0]][predicted_label_list[0]] += 1

            actual_data.extend(labels.tolist())  # actual_data : index 번째 데이터 정답
            predict_data.extend(predicted.tolist())  # predict_data : index 번째 예측 값

            image_path_list.extend(image_path)  # index 번째 image_path

    return total_correct, actual_data, predict_data, image_path_list


# log 경로 지정
def setSysoutpath(ex_path):
    if not os.path.isdir(f'./{ex_path}'):
        os.makedirs(f'./{ex_path}')
    sys.stdout = open(f'./{ex_path}/output_{ex_path}.csv', 'w', encoding='utf8')


# best_model 가중치 저장하기
def save_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


# scikit-learn f1 score 구하기
def sklearnF1Score(y_pred_list, my_data, model_name):
    y_pred_list = [a for a in y_pred_list]
    my_data = [a for a in my_data]

    my_data = torch.tensor(my_data)
    y_pred_list = torch.tensor(y_pred_list)

    my_data = torch.flatten(my_data)
    y_pred_list = torch.flatten(y_pred_list)
    f1_score = classification_report(my_data, y_pred_list)
    print(f"***************{model_name} F-1 Score*******************")
    print("")
    print(f1_score)


# F1 Score 구하기
def getF1Score(predict_list, actual_data, model_name, actual_class, num_classes, class_names):
    sklearnF1Score(predict_list, actual_data, model_name)
    Average_precision, Average_recall, Accuracy, F1_Score = confusion_matrix(actual_class, num_classes,
                                                                             class_names)
    return F1_Score


# GT 구하기
def getGroundTruth(ground_list, predicted_list, image_path_list):
    for u in range(len(ground_list)):
        now = time.localtime()
        print(
            f"[{now.tm_year}/{now.tm_mon}/{now.tm_mday} {now.tm_hour}:{now.tm_min}/{now.tm_sec}] the result >>> ground_truth : {ground_list[u]}, predicted : {predicted_list[u]}, image_path : {image_path_list[u]}")


# VGG16 Algorithm
def runVgg16(train_loader, validation_loader, train_set, validation_set):
    model = models.vgg16(pretrained=True)

    num_features = model.classifier[6].in_features
    # for classifier in model.classifier :
    #    print(classifier)
    actual_class = [[0 for j in range(6)] for k in range(6)]
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, 6)])
    model.classifier = nn.Sequential(*features)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    run(train_loader, validation_loader, device, optimizer, net, criterion, scheduler, train_set, validation_set, 6,
        "vgg16", actual_class, "CITRUS VGG")


# Resnet50 Algorithm
def runResnet(train_loader, validation_loader, train_set, validation_set):
    model = models.resnet50(pretrained=True)

    # change the output layer to 10 classes
    num_classes = 6
    actual_class = [[0 for j in range(num_classes)] for k in range(num_classes)]
    num_classes = train_set.num_classes
    class_names = train_set.class_names
    print(class_names)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    run(train_loader, validation_loader, device, optimizer, net, criterion, scheduler, train_set, validation_set,
        num_classes, "resnet50", actual_class, "CITRUS RESNET")


# Densenet161 Algorithm
def runDensenet(train_loader, validation_loader, train_set, validation_set):
    model = models.densenet161(pretrained=True)

    # change the output layer to 10 classes
    num_classes = 6
    actual_class = [[0 for j in range(num_classes)] for k in range(num_classes)]
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    run(train_loader, validation_loader, device, optimizer, net, criterion, scheduler, train_set, validation_set,
        num_classes, "densenet", actual_class, "CITRUS DENSENET")


# Efficientnet Algorithm
def runEfficientNEt(train_loader, validation_loader, train_set, validation_set):
    num_classes = 6
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)

    # change the output layer to 10 classes

    actual_class = [[0 for j in range(num_classes)] for k in range(num_classes)]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    run(train_loader, validation_loader, device, optimizer, net, criterion, scheduler, train_set, validation_set,
        num_classes, "efficientnet", actual_class, "CITRUS EfficientNEt")


# VIT Algorithm
def runVIT(train_loader, validation_loader, train_set, validation_set):
    num_classes = 6
    model = ViT('B_16_imagenet1k', pretrained=True, num_classes=num_classes, image_size=224)

    # change the output layer to 10 classes

    actual_class = [[0 for j in range(num_classes)] for k in range(num_classes)]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    run(train_loader, validation_loader, device, optimizer, net, criterion, scheduler, train_set, validation_set,
        num_classes, "vit", actual_class, "CITRUS VIT")


def main():
    trans_train = transforms.Compose([transforms.Resize((224, 224)),  # 224 x 224 size 로 resize
                                      transforms.RandomHorizontalFlip(p=0.5),  # 이미지 좌우를 50%의 확률로 반전
                                      transforms.RandomRotation(degrees=(-90, 90)),
                                      transforms.RandomVerticalFlip(p=0.5),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(degrees=(-90, 90)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trans_validation = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.RandomRotation(degrees=(-90, 90)),
                                           transforms.RandomVerticalFlip(p=0.5),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(degrees=(-90, 90)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data_set = CustomImageDataset(
        data_set_path="E:/Fruits/dataset/Training/감귤/Training",
        transforms=trans_train)
    val_data_set = CustomImageDataset(
        data_set_path="E:/Fruits/dataset/Validation/감귤/Validation",
        transforms=trans_validation)

    train_loader = DataLoader(train_data_set, num_workers=2, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data_set, num_workers=2, batch_size=64, shuffle=True)

#    runVgg16(train_loader, val_loader, train_data_set, val_data_set)
#    runResnet(train_loader, val_loader, train_data_set, val_data_set)
#    runDensenet(train_loader, val_loader, train_data_set, val_data_set)
#    runEfficientNEt(train_loader, val_loader, train_data_set, val_data_set)
    runVIT(train_loader, val_loader, train_data_set, val_data_set)


if __name__ == '__main__':
    main()
