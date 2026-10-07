# Automação de Precificação de Postos (utilizando YOLOv8 + EasyOCR)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-darkblue.svg)](https://pandas.pydata.org/)
[![LGPD Compliant](https://img.shields.io/badge/LGPD-Compliant-success.svg)]()

Um projeto já aplicado dentro da empresa que trabalho de **Visão Computacional e Engenharia de Dados** desenhado para automatizar a extração de preços de combustíveis a partir de fotografias de totens de postos concorrentes e clientes. 

Este projeto substitui a digitação manual por um sistema inteligente de detecção, auditoria de variação financeira e atualização dinâmica de planilhas de mercado.
Sendo produzido desde o treinamento das redes neurais convulacionais(CNN´s) dentro de Yolo e EasyOCR(sendo essas com função de ativação Silu e Sigmoide) até a organização das pastas.

---

## 🧠 Arquitetura e Regras de Negócio (Diferenciais)

Foi construído com foco em **confiabilidade de dados e preenchimento de dados de forma automatizada**, aplicando lógicas de tratamento(limitadores) antes de qualquer gravação no banco de dados (Excel).

*   **Detecção Dupla (YOLOv8 + EasyOCR):** O YOLOv8 localiza espacialmente as coordenadas (Bounding Boxes - "caixinhas") dos tipos de combustíveis e das zonas de preços em que o modelo foi treinado customizadamente para totens de preços. O EasyOCR entra em seguida apenas nas zonas delimitadas para extrair o valor monetário já que ele age como um ótimo captador de textos.
*   **Pareamento Espacial Algorítmico:** Utiliza *Linear Sum Assignment* (`scipy.optimize`) para calcular a distância nos eixos X e Y entre o nome do combustível e o número detectado, garantindo que o preço lido pertence à gasolina correta, e não ao diesel da linha de baixo.
*   **A "Trava Dupla" (Double Lock Validation):** O sistema só autoriza a gravação se houver um encontro coerente e conciso entre a foto analisada e o cadastro na base de dados, validando o par **Posto + Cidade (Região)**.
*   **Auditoria de Variação de Preço:** Uma camada de defesa em que sistema consulta o histórico daquele posto no dia anterior, baseado numa prior genérica. Logo, se a IA ler um preço que represente uma variação abrupta (ex: erro de leitura OCR gerando um salto > 5%), a atualização é **bloqueada e rejeitada** para evitar corrupção da base de inteligência.
*   **LGPD e Confidencialidade:** Para fins de portfólio público, **este repositório foi higienizado** ,ou seja, os nomes de clientes reais e bases confidenciais foram substituídas por nomes genéricos e dados anonimizados e também foi retirado as fotos utilizadas e como o modelo foi feito.

---

## ⚙️ Funcionalidades

- [x] Interface de Linha de Comando interativa para seleção rápida de zonas e regiões.
- [x] Processamento em lote de imagens (busca sempre a fotografia mais recente do dia para cada posto).
- [x] Correção de ruídos de leitura de OCR (filtros Regex e normalização Unicode).
- [x] Gestão autônoma de calendário: O script clona automaticamente o _layout_ da planilha do dia anterior caso hoje seja um novo dia de pesquisa.

---

## 📁 Estrutura do Projeto

```text
projeto-precificacao-postos/
│
├── data/                        #Diretório de dados 
│   ├── base_precos.xlsx        #Planilha modelo (Anonimizada)
│   ├── fotos_postos/           #Pastas divididas por Regiões e Postos
│   └── fotos_postos_clientes/  #Estrutura espelhada para o alvo interno
│
├── modelo/                     #Arquivos do modelo de Deep Learning
│   └── runs/detect/weights/
│       └── best.pt             #Pesos treinados do YOLOv8 (Custom)
│
├── script/
│   └── automacao_placas.py     #Script principal 
│      └── remove_fotos.py     #Remove as fotos referentes aos postos
│
├── requirements.txt            #Dependências do projeto
└── README.md
