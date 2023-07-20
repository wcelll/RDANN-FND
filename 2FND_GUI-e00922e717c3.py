import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog, \
    QRadioButton, QHBoxLayout, QPlainTextEdit, QMessageBox
from PyQt5.QtGui import QPixmap
from RDANN import CNN_Fusion  # 确保CNN_Fusion类在你的代码中是可用的
import torch
import numpy as np
from transformers import BertTokenizer
from RDANN import CNN_Fusion  # 确保CNN_Fusion类在你的代码中是可用的
import argparse
from PIL import Image
from torchvision import transforms


# Create device object
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class App(QWidget):

    def __init__(self, main_func):
        super().__init__()
        self.title = '基于PyQt5的多模态虚假新闻识别系统'
        self.main_func = main_func
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.layout = QVBoxLayout()

        self.model_button = QPushButton('选择模型文件', self)
        self.model_button.clicked.connect(self.select_model)
        self.layout.addWidget(self.model_button)

        self.radio_text = QRadioButton("文本输入")
        self.radio_image = QRadioButton("图片输入")
        self.radio_text.setChecked(True)

        hbox_radio = QHBoxLayout()
        hbox_radio.addWidget(self.radio_text)
        hbox_radio.addWidget(self.radio_image)
        self.layout.addLayout(hbox_radio)

        self.text_input = QPlainTextEdit(self)
        self.layout.addWidget(self.text_input)
        self.image_button = QPushButton('选择图片文件', self)
        self.image_button.clicked.connect(self.select_image)
        self.layout.addWidget(self.image_button)

        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        self.submit_button = QPushButton('开始预测', self)
        self.submit_button.clicked.connect(self.start_prediction)
        self.layout.addWidget(self.submit_button)

        self.result_label = QLabel(self)
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)

    def select_model(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, '选择模型文件', '', 'Pickle Files (*.pkl)', options=options)
        if file_name:
            self.main_func.args.model_path = file_name

    def select_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, '选择图片文件', '', 'Images (*.png *.xpm *.jpg *.bmp)',
                                                   options=options)
        if file_name:
            self.main_func.args.image_path = file_name
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(224, 224))

    def start_prediction(self):
        if self.radio_text.isChecked():
            self.main_func.args.input_type = 'text'
            self.main_func.args.text_input = self.text_input.toPlainText()
            if not self.main_func.args.text_input:
                QMessageBox.warning(self, "警告", "请输入文本！")
                return
        elif self.radio_image.isChecked():
            self.main_func.args.input_type = 'image'
            if not hasattr(self.main_func.args, 'image_path'):
                QMessageBox.warning(self, "警告", "请选择图片！")
                return

        prediction = self.main_func.predict_wrapper()
        if prediction == 0:
            result = '真实新闻'
        else:
            result = '虚假新闻'

        self.result_label.setText(f'预测结果：{result}')


def parse_arguments(parser):
    parser.add_argument('training_file', type=str, metavar='<training_file>', help='')
    # parser.add_argument('validation_file', type=str, metavar='<validation_file>', help='')
    parser.add_argument('testing_file', type=str, metavar='<testing_file>', help='')
    parser.add_argument('output_file', type=str, metavar='<output_file>', help='')
    parse.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default=32, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--filter_num', type=int, default=5, help='')
    parser.add_argument('--lambd', type=int, default=1, help='')
    parser.add_argument('--text_only', type=bool, default=False, help='')

    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--num_epochs', type=int, default=300, help='')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='')
    parser.add_argument('--event_num', type=int, default=10, help='')

    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
    parser.add_argument('--input_type', type=str, default='text', choices=['text', 'image'],
                        help='Choose input type: text or image')

    #    args = parser.parse_args()
    return parser


def main_qt(args):
    app = QApplication(sys.argv)
    main_func = MainFunc(args)
    ex = App(main_func)
    ex.show()
    sys.exit(app.exec_())


def load_model(model_path, args):
    model = CNN_Fusion(args)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_text(text, args):
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    tokenized_text = tokenizer.encode(text)
    input_text = []
    mask_seq = np.zeros(args.sequence_length, dtype=np.float32)
    mask_seq[:len(tokenized_text)] = 1.0
    while len(tokenized_text) < args.sequence_length:
        tokenized_text.append(0)
    input_text.append(tokenized_text)
    input_mask = [mask_seq]
    return input_text, input_mask


def predict(model, text, mask):
    model.eval()
    with torch.no_grad():
        text = torch.tensor(text, dtype=torch.long).view(1, -1).to(device)
        mask = torch.tensor(mask, dtype=torch.float32).view(1, -1).to(device)
        empty_image_input = torch.zeros((1, 3, 224, 224), dtype=torch.float32).to(device)
        outputs, _ = model(text, empty_image_input, mask)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

def preprocess_image(image_path, args):
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    processed_image = transform(image)
    processed_image = processed_image.unsqueeze(0).to(device)

    return processed_image

def predict_image(model, image):
    model.eval()
    with torch.no_grad():
        empty_text_input = torch.zeros((1, args.sequence_length), dtype=torch.long).to(device)
        empty_mask_input = torch.zeros((1, args.sequence_length), dtype=torch.float32).to(device)
        image = image.to(device)
        outputs, _ = model(empty_text_input, image, empty_mask_input)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()


class MainFunc:
    def __init__(self, args):
        self.args = args

    def predict_wrapper(self):
        # 加载训练好的模型
        model = load_model(self.args.model_path, self.args)

        if self.args.input_type == 'text':
            # 输入新闻文本
            news_text = self.args.text_input

            # 预处理文本
            input_text, input_mask = preprocess_text(news_text, self.args)
            # 使用模型进行预测
            prediction = predict(model, input_text, input_mask)

        elif self.args.input_type == 'image':
            # 输入新闻图片
            news_image = self.args.image_path

            # 预处理图片
            input_image = preprocess_image(news_image, self.args)

            # 使用模型进行预测
            prediction = predict_image(model, input_image)

        return prediction


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    train = ''
    test = ''
    output = 'D:/RDANN-FND/Data/weibo/output_image_text'
    args = parser.parse_args([train, test, output])

    main_qt(args)







