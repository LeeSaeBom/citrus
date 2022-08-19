import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import time
import os
import sys
from customdataset_3 import CustomImageDataset
from confusion_matrix_210825 import confusion_matrix
from efficientnet_pytorch import EfficientNet
from pytorch_pretrained_vit import ViT


# test_loader : 테스트 데이터 불러오기
# device : cuda
# net : model
# test_set : test data
# num_classes : class 갯수
# model_name : Resnet50, Densenet161
# actual_class : confusion matrix 배열
# ex_path : log 저장 path

def run(test_loader, device, net, test_set, num_classes, model_name, actual_class, ex_path):
    setSysoutPath(ex_path)  # 표준 출력을 파일 출력으로 변경
    now = time.localtime()  # 현재 시간
    print("Start test.py")
    print("[%04d/%02d/%02d %02d:%02d:%02d] 감귤 %s Test_start" % (
        now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec, model_name))
    net.eval()  # 사용하지 않아야 하는 layer off

    class_names = test_set.class_names
    print(f"class names: {class_names}")

    # 모델 데이터 검증
    total_correct, actual_data, predicted, image_path_list = process(test_loader,
                                                                     device,
                                                                     net,
                                                                     actual_class)

    # 정확도 계산
    e_acc = total_correct / len(test_set)

    # GT 값 계산
    getGroundTruth(actual_data, predicted, image_path_list)

    # F1 Score 계산
    f1_score = getF1Score(predicted, actual_data, model_name, actual_class, num_classes, class_names)

    print('')
    print('Finished Test')
    print(f'Best {model_name} Acc :', e_acc)
    print(f'Best F1 score :', f1_score)

    end = time.localtime()
    print("[%04d/%02d/%02d %02d:%02d:%02d] Test_finished" % (
        end.tm_year, end.tm_mon, end.tm_mday, end.tm_hour, end.tm_min, end.tm_sec))
    sys.stdout.flush()


# log 경로 지정
def setSysoutPath(ex_path):
    if not os.path.isdir(f'./{ex_path}'):
        os.makedirs(f'./{ex_path}')
    sys.stdout = open(f'./{ex_path}/output_{ex_path}_test_log.csv', 'w', encoding='utf8')


# 모델 데이터 검증
def process(test_loader, device, net, actual_class):
    with torch.no_grad():  # with torch.no_grad() 메모리 사용량을 줄이고 연산 속도를 높이기 위해 사용
        total_correct = 0  # 정확도
        loop = tqdm(enumerate(test_loader), total=len(test_loader))
        image_path_list = []  # 이미지 path list
        actual_data = []  # 정답 list
        predict_data = []  # 예측 list

        for i, data in loop:
            inputs = data['image'].to(device)
            labels = data['label'].to(device)
            image_path = data['image_path']

            outputs = net(inputs)

            _, predicted = torch.max(outputs, 1)

            test_correct = torch.sum(predicted == labels.data).item()

            total_correct += test_correct

            labels_list = labels.tolist()
            predicted_label_list = predicted.tolist()

            actual_class[labels_list[0]][predicted_label_list[0]] += 1

            actual_data.extend(labels.tolist())  # actual_data : index 번째 데이터 정답
            predict_data.extend(predicted.tolist())  # predict_data : index 번째 예측 값

            image_path_list.extend(image_path)  # index 번째 image_path

    return total_correct, actual_data, predict_data, image_path_list


# F1 Score 구하기
def getF1Score(predict_list, actual_data, model_name, actual_class, num_classes, class_names):
    sklearnF1Score(predict_list, actual_data, model_name)
    _, _, _, f1_score = confusion_matrix(actual_class, num_classes, class_names)
    return f1_score


# GT 구하기
def getGroundTruth(ground_list, predicted_list, image_path_list):
    for u in range(len(ground_list)):
        now = time.localtime()  # 현재 시간
        print(
            f"[{now.tm_year}/{now.tm_mon}/{now.tm_mday} {now.tm_hour}:{now.tm_min}:{now.tm_sec}] the result >>> ground_truth : {ground_list[u]}, predicted : {predicted_list[u]}, image_path : {image_path_list[u]}")


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

# VGG16 Algorithm
def runVgg16(testloader, test_set):
    model = models.vgg16(pretrained=True)

    num_features = model.classifier[6].in_features
    # for classifier in model.classifier :
    #    print(classifier)
    actual_class = [[0 for j in range(6)] for k in range(6)]
    features = list(model.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, 6)])
    model.classifier = nn.Sequential(*features)

    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model.load_state_dict : 훈련된 가중치 모델 불러오기
    model.load_state_dict(torch.load("best_vgg16-95.pth", map_location=device_name))
    device = torch.device(device_name)

    net = model.to(device)

    run(testloader, device, net, test_set, 6, "vgg16", actual_class, "CITRUS VGG TEST")



# Resnet Algorithm
def runResnet(testloader, test_set):
    model = models.resnet50(pretrained=True)

    # change the output layer to 10 classes
    num_classes = 6
    actual_class = [[0 for j in range(num_classes)] for k in range(num_classes)]
    num_classes = test_set.num_classes
    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, num_classes)

    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model.load_state_dict : 훈련된 가중치 모델 불러오기
    model.load_state_dict(torch.load("best_resnet50-99.pth", map_location=device_name))
    device = torch.device(device_name)

    net = model.to(device)

    run(testloader, device, net, test_set, num_classes, "resnet50", actual_class, "CITRUS RESNET TEST")


# Densenet Algorithm
def runDensenet(testloader, test_set):
    model = models.densenet161(pretrained=True)

    # change the output layer to 10 classes
    num_classes = 6
    actual_class = [[0 for j in range(num_classes)] for k in range(num_classes)]
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model.load_state_dict : 훈련된 가중치 모델 불러오기
    model.load_state_dict(torch.load("best_densenet-97.pth", map_location=device_name))
    device = torch.device(device_name)

    net = model.to(device)

    run(testloader, device, net, test_set, num_classes, "densenet", actual_class, "CITRUS DENSENET TEST")


# EfficientNet Algorithm
def runEfficientNet(testloader, test_set):
    num_classes = 6
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)

    # change the output layer to 10 classes
    actual_class = [[0 for j in range(num_classes)] for k in range(num_classes)]

    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model.load_state_dict : 훈련된 가중치 모델 불러오기
    model.load_state_dict(torch.load("best_efficientnet-95.pth", map_location=device_name))
    device = torch.device(device_name)

    net = model.to(device)

    run(testloader, device, net, test_set, num_classes, "efficientnet", actual_class, "CITRUS EfficientNEt TEST")


# VIT Algorithm
def runVIT(testloader, test_set):
    num_classes = 6
    model = ViT('B_16_imagenet1k', pretrained=True, num_classes=num_classes, image_size=224)

    # change the output layer to 10 classes
    actual_class = [[0 for j in range(num_classes)] for k in range(num_classes)]

    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model.load_state_dict : 훈련된 가중치 모델 불러오기
    model.load_state_dict(torch.load("best_vit-60.pth", map_location=device_name))
    device = torch.device(device_name)

    net = model.to(device)

    run(testloader, device, net, test_set, num_classes, "vit", actual_class, "CITRUS VIT TEST")


def main():
    trans_test = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_data_set = CustomImageDataset(
        data_set_path="/root/SB/dataset/Test/Citrus",
        transforms=trans_test)

    test_loader = DataLoader(test_data_set, num_workers=2, batch_size=1, shuffle=True)
    runVgg16(test_loader,test_data_set)
    runResnet(test_loader, test_data_set)
    runDensenet(test_loader, test_data_set)
    runEfficientNet(test_loader,test_data_set)
    runVIT(test_loader,test_data_set)


if __name__ == '__main__':
    main()
