import pandas as pd
import matplotlib.pyplot as plt

# 读取csv文件
df = pd.read_csv('D:/code/data.csv')

# 绘制Loss曲线
plt.plot(df['Epoch'], df['Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.savefig('Loss.png')
plt.show()

# 绘制Class Loss曲线
plt.plot(df['Epoch'], df['Class Loss'])
plt.xlabel('Epoch')
plt.ylabel('Class Loss')
plt.title('Class Loss vs. Epoch')
plt.savefig('Class_Loss.png')
plt.show()

# 绘制Domain Loss曲线
plt.plot(df['Epoch'], df['domain loss'])
plt.xlabel('Epoch')
plt.ylabel('domain loss')
plt.title('domain loss vs. Epoch')
plt.savefig('domain_loss.png')
plt.show()

# 绘制Train Acc曲线
plt.plot(df['Epoch'], df['Train_Acc'])
plt.xlabel('Epoch')
plt.ylabel('Train_Acc')
plt.title('Train Acc vs. Epoch')
plt.savefig('Train_Acc.png')
plt.show()

# 绘制Validate Acc曲线
plt.plot(df['Epoch'], df['Validate_Acc'])
plt.xlabel('Epoch')
plt.ylabel('Validate_Acc')
plt.title('Validate Acc vs. Epoch')
plt.savefig('Validate_Acc.png')
plt.show()
