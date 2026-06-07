# 🐾 Pet Segmentation App (Android + PyTorch)

Este projeto implementa uma aplicação Android nativa capaz de realizar **Segmentação Semântica** de pets (gatos e cachorros) em tempo real, utilizando um modelo de Inteligência Artificial treinado no PyTorch e otimizado para dispositivos móveis através do Google LiteRT (TensorFlow Lite).

O aplicativo permite ao usuário selecionar uma imagem da galeria do smartphone e aplicar a inferência, gerando uma máscara translúcida sobreposta à foto original para destacar o animal, ignorando o fundo.

## 🚀 Arquitetura e Tecnologias

### Inteligência Artificial (Modelo)
* **Framework:** PyTorch
* **Arquitetura Base:** U-Net Customizada (`UNetTransferMobileNetV2`)
* **Backbone (Transfer Learning):** MobileNetV2 (pré-treinado no ImageNet)
* **Dataset:** Oxford-IIIT Pet Dataset (com normalização alinhada ao app mobile)
* **Exportação:** PyTorch para TFLite via infraestrutura **LiteRT** (`litert-torch`).

### Engenharia de Software (Mobile)
* **Plataforma:** Android Nativo
* **Linguagem:** Kotlin
* **Motor de Inferência:** TensorFlow Lite (`org.tensorflow:tensorflow-lite:2.16.1`)
* **Tratamento de Dados:** Motor universal com leitura direta de memória via `ByteBuffer`, compatível com saídas dinâmicas **NCHW** e **NHWC** para evitar problemas de compatibilidade e corrupção de tensores.

## 📊 Resultados do Treinamento

O modelo foi treinado por 25 épocas utilizando o dataset original, aproveitando o poder do Transfer Learning para extração de features (*encoder* congelado) e treinando o *decoder* da U-Net para a tarefa específica de segmentação binária focada em bordas e preenchimento.

**Métricas Finais (Época 25):**
* 🎯 **Acurácia (Macro):** 89.60% (`0.8960`)
* 📐 **IoU (Intersection over Union):** 82.93% (`0.8293`)
* 📉 **Loss (CrossEntropy):** `0.1251`

## 🛠️ Como Executar o Projeto

### 1. Testando o Aplicativo Android
1.  Clone este repositório.
2.  Abra a pasta do projeto no **Android Studio**.
3.  Aguarde o Gradle realizar a sincronização (Sync).
4.  Certifique-se de que o arquivo `model.tflite` está presente na pasta `app/src/main/assets/`.
5.  Conecte seu smartphone físico via Depuração USB ou inicie um Emulador.
6.  Clique em **Run** (`Shift + F10`) ou gere o APK via `Build > Generate APKs`.

### 2. Treinando o Modelo (Opcional)
O notebook de treinamento completo (`Android_segmentation.ipynb`) está incluído na raiz deste repositório. Para reproduzir o treinamento ou realizar novos testes:
1.  Faça o upload do arquivo `.ipynb` para o **Google Colab**.
2.  No Colab, altere o tempo de execução para utilizar uma **GPU (T4 ou superior)**.
3.  Execute todas as células sequencialmente. O último bloco exportará um novo arquivo `model.tflite` pronto para ser consumido pelo Kotlin.

## 📱 Demonstração Visual

Abaixo, os resultados da inferência rodando nativamente no smartphone com múltiplas imagens de testes reais (cachorros e gatos):

<img width="720" height="1600" alt="segmentacao-dog1" src="https://github.com/user-attachments/assets/fc34c0a4-4b73-4970-8c19-04536bbbbd2d" />
<img width="720" height="1600" alt="segmentacao-cat1" src="https://github.com/user-attachments/assets/7638afa6-7a96-4cbc-b057-f3b49991ce99" />
<img width="720" height="1600" alt="segmentacao-cat2" src="https://github.com/user-attachments/assets/8b564028-b733-4ad4-a1e4-682141e178ea" />

---
*Projeto desenvolvido como parte da atividade prática de Segmentação Semântica e Implantação.*
