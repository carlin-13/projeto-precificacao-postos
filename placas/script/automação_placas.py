#================
#0)- Bibliotecas utilizadas
#================

import cv2  #biblioteca opencv para manipulação de imagens
import re   #biblioteca para trabalhar com expressões regulares, nas funções de limpeza
import os   #biblioteca para manipulação de arquivos
import glob #biblioteca para manipulação de arquivos e diretorios
import numpy as np #biblioteca para operações matriciais
import unicodedata #biblioteca para normalização de caracteres
import easyocr  #biblioteca para visão computacional por rede neural (OCR)
import pandas as pd #biblioteca para organização dos arquivos, leitura e escrita
from scipy.optimize import linear_sum_assignment #biblioteca para pareamento de deteções, ou seja, combinação de deteções de combustível e preços
from ultralytics import YOLO  #biblioteca para visão computacional por rede neural (YOLO), mais focada na detecção de objetos
import datetime #adicionar biblioteca datetime para pegar a data atual

#==============================
#1)Caminhos e arquivos locais
#==============================

#Caminhos

Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pasta_base_fotos = os.path.join(Base_dir, "data","fotos_postos") 
caminho_planilha = os.path.join(Base_dir, "data","base.xlsx")
caminho_modelo = os.path.join(Base_dir,"modelo","runs","detect","train","weights","best.pt")
pasta_cliente = os.path.join(Base_dir, "data","fotos_postos_clientes")

#==========================================
#2)-Configurações de OCR e YOLO
#==========================================
#Configurações do modelo 

leitor_ocr = easyocr.Reader(['pt', 'en'], gpu=False) #Gpu desativada devido falta desse hardwarwe na minha máquina
modelo = YOLO(caminho_modelo)

#Colunas que serão usadas

col_posto = 'POSTO_CONCORRENTE'
col_combustivel = 'COMBUSTÍVEL'
col_preco_concorrente = 'PRECO_CONCORRENTE' #coluna que vai ser atualizada
col_data = 'DATA'
col_preco_cliente = 'PRECO_CLIENTE_ALVO' #coluna utilizada para atualizar o preço do ponteio 
col_posto_cliente = 'POSTOS_CLIENTE'
col_cidade = "CIDADE"

#Limiares

#exemplo definido de forma genérica

LIMIAR_YOLO = 0.30
LIMIAR_OCR = 0.40
LIMIAR_VOTO_OCR = 0.40
LIMITE_VARIACAO_PRECO = 0.10

#==========================================
#3)- Funções auxiliares(normalizar_texto, aumentar_caixa, centro, parear_deteccoes)
#==========================================

#Função para normalizar o texto
def normalizar_texto(texto):
    if pd.isna(texto) or texto is None:
        return "" #laço condicional se é um NA e se for returnar nada
    texto = str(texto).upper()        #todo texto sera em maiusculo
    texto = unicodedata.normalize('NFKD', texto)   #Normalizando os caracteres para evitar acentos e outros problemas de OCR
    texto = "".join(c for c in texto if not unicodedata.combining(c))  #Removendo acentos
    texto = re.sub(r'[^A-Z0-9 ]', '', texto)  #removendo caracteres especiais
    return re.sub(r"\s+", " ", texto).strip() #removendo espaços duplos


#Função para criar uma chave rigorosa de comparação.
def normalizar_chave(texto):
    texto_normalizado = normalizar_texto(texto)   #texto normalizado aplicado na função texto 
    return re.sub(r'[^A-Z0-9]', '', texto_normalizado)  #retorna o padrão da chave com apenas letras e numeros


#Função para normalizar datas 
#mantendo o padrão DD/MM/AAAA
def normalizar_data(valor):
    if valor is None or pd.isna(valor):   
        return pd.NaT   #se valor for nulo ou nulo, retorna nulo
    if isinstance(valor, (pd.Timestamp, datetime.datetime, datetime.date)):
        return pd.Timestamp(valor).normalize()    #se for uma data, retorna a mesma
    #Tratamento adicional para eventual número serial de data vindo do Excel
    if isinstance(valor, (int, float, np.integer, np.floating)):
        if 20000 <= float(valor) <= 80000:
            return (pd.Timestamp('1899-12-30') + pd.to_timedelta(float(valor), unit='D')).normalize()
    texto = str(valor).strip()  #texto para ser tratado o texto da data
    data = pd.to_datetime(texto, dayfirst=True, errors='coerce')   #opd.to_datetime transforma o texto em data
    if pd.isna(data):
        return pd.NaT
    return pd.Timestamp(data).normalize()


#Função para aumentar a caixa delimitadora da detecção
def aumentar_caixa(box, imagem, pad_x=0.04, pad_y=0.10):  
    x1, y1, x2, y2 = box    #desestruturando o box
    h_img, w_img = imagem.shape[:2]  #pegando as dimensões da imagem
    largura, altura = x2 - x1, y2 - y1  #pegando as dimensões do box
    x1, x2 = max(0, int(x1 - largura * pad_x)), min(w_img, int(x2 + largura * pad_x)) #aumento da margem em x 
    y1, y2 = max(0, int(y1 - altura * pad_y)), min(h_img, int(y2 + altura * pad_y)) #aumento da margem em y
    return imagem[y1:y2, x1:x2]  #retorna a imagem recortada com a caixa aumentada


#Função para calcular o centro da caixa delimitadora
def centro(box):
    return (int((box[0]+box[2])/2), int((box[1]+box[3])/2)) #retorna o centro da caixa


#Função para parear as detecções de combustíveis e preços
def parear_deteccoes(combustiveis, precos, largura_imagem, altura_imagem):
    if not combustiveis or not precos: return []   #Laço condicional de se as listas estiverem vazias
    custos = np.zeros((len(combustiveis), len(precos)))  #variavel custos que vai armazenar os custos de pareamento entre combustíveis e preços
    for i, comb in enumerate(combustiveis):     #laço para percorrer a lista de combustíveis
        cx_comb, cy_comb = centro(comb["box"])   #calculando o centro do combustível
        for j, preco in enumerate(precos):     #laço para percorrer a lista de preços
            cx_preco, cy_preco = centro(preco["box"])  #calculando o centro do preco
            dy, dx = (abs(cy_comb - cy_preco) / altura_imagem), (abs(cx_comb - cx_preco) / largura_imagem)  #calculando a distância vertical e horizontal
            penalidade_lado = 0.15 if cx_preco < cx_comb else 0  #penalidade para preços que estão do lado esquerdo do combustível
            custos[i, j] = (dy * 5.0 + dx * 0.40 + penalidade_lado)  #calculando o custo
    linhas, colunas = linear_sum_assignment(custos)  #encontrando as linhas e colunas para pareamento
    pares = []  #lista para armanezar os pares de combust[íveis e preços
    for i, j in zip(linhas, colunas):  #laço para percorrer as linhas e colunas
        if custos[i, j] > 0.35: continue   #se o custo for maior que 0.35, não faz o pareamento
        cy_comb, cy_preco = centro(combustiveis[i]["box"])[1], centro(precos[j]["box"])[1] #calculando o centro do combustível e do preço
        if (abs(cy_comb - cy_preco) / altura_imagem) > 0.18: continue #se a distância vertical for maior que 0.18, não faz o pareamento
        pares.append((combustiveis[i], precos[j], custos[i, j])) #adicionando o par de combustível e preço na lista
    return pares  #retorna a lista de pares de combustíveis e preços



#mapa de números, para ser um limitador de numeros que possam ser confundidos

MAPA_NUMEROS = str.maketrans({"O":"0", "Q":"0", "D":"0", "I":"1", "L":"1", "Z":"2", "S":"5", "G":"6", "B":"8"}) #dicionario para que se confundir as letras com os numeros ele identificar como número



#Função para transformar o texto em preço
def transformar_em_preco(texto):
    if not texto:
        return None
    texto = texto.upper().translate(MAPA_NUMEROS)   #o texto é colocado em maiusculos e traduzindo as letras para numeros
    digitos = re.sub(r"\D", "", texto)  #removendo todos os caracteres que nao sejam numeros
    if len(digitos) not in (3, 4):
        return None     #se o comprimento for diferente de 3 ou 4, retorna nulo
    try:
        valor = int(digitos) / 100  #transformando os numeros em preços e dividindo por 100 para ter um preço real
    except ValueError:
        return None
    #Corrige erro comum do OCR:
    if 10.00 < valor < 20.00:  #se o leitor voltar um preço muito alto ou muito baixo
        valor_corrigido = valor - 10  #subtraindo 10
        if 5.00 <= valor_corrigido <= 9.99:   #se o preço corrigido estiver entre 5 e 10
            print(f"🔧 Correção OCR aplicada: R$ {valor:.2f} -> R$ {valor_corrigido:.2f}")
            valor = valor_corrigido
    #Filtro de segurança
    if not 3.00 <= valor <= 10.00:  #se o preço for menor que 3 ou maior que 10
        print(
            f"⚠️ Preço descartado pelo filtro de valor: "
            f"R$ {valor:.2f}"
        )
        return None
    return round(valor, 2) #retorna o preço com 2 casas decimais


#função para ler o preço usando OCR
def ler_preco(recorte):
    try:
        #Modo simples e direto para rodar rápido na máquina local
        resultados = leitor_ocr.readtext(recorte, allowlist="0123456789,.-R$ ", detail=1) #retorna o texto e a confianca
    except Exception:  #laço condicional se ocorrer um erro
        return None, 0.0 #retorna o texto e a confianca
    melhor_preco, melhor_confianca = None, 0.0 #variaveis para armazenar o melhor texto e confianca
    for res in resultados:   #laço para percorrer os resultados
        _, texto, conf = res #desestruturando o resultado
        if conf < LIMIAR_VOTO_OCR: continue #laço condicional se a confianca for menor que o limiar
        preco = transformar_em_preco(texto)  #variavel para armazenar o preço
        if preco and conf > melhor_confianca:  #laço condicional se o preço for válido e a confianca for maior que a melhor confianca
            melhor_preco = preco
            melhor_confianca = conf
    
    return melhor_preco, melhor_confianca   #retorna o melhor preço e a melhor confianca





#===========================
#4)- Função para processamento da foto 
#===========================

#Função para processar a foto
def buscar_preco_anterior(df, posto, combustivel, data_atual,coluna_alvo, coluna_chave_posto,cidade):
    mascara_posto = (
        df[coluna_chave_posto].apply(normalizar_chave) ==   #normalizando as chaves do posto para comparação de chaves auxiliares e ter a base de quem sera comparado
        normalizar_chave(posto)
    )

    mascara_comb = (
        df["_CHAVE_COMB"].apply(normalizar_chave) ==  #normalizando as chaves do combustivel para comparação de chaves auxiliares e ter a base de quem sera comparado
        normalizar_chave(combustivel)
    )

    mascara_cidade = (
        df["_CHAVE_CIDADE"].apply(normalizar_chave) == 
        normalizar_chave(cidade)
    )

    datas_validas = df[   #mostra as datas que já se tem dentro da base de dados
        mascara_posto & 
        mascara_comb &
        mascara_cidade &
        (df[col_data] < data_atual) &
        df[coluna_alvo].notna()    #coluna_alvo dinamica
    ]
    print(
    f"🔎 Histórico encontrado: {posto} | {combustivel} |  "    #Mostra o histórico 
    f"{len(datas_validas)} registro(s)"
)
    if datas_validas.empty:    #se nenhuma data for encontrada
        return None
    ultima_linha = datas_validas.sort_values(
        col_data,
        ascending=False  #ordena as datas em ordem decrescente
    ).iloc[0]
    try:
        return float(     
            str(ultima_linha[coluna_alvo]).replace(",", ".")  #se nenhuma data for encontrada
        )
    except (ValueError, TypeError):
        return None


#Função para processar a foto
def processar_foto(caminho_img, nome_posto, df_planilha, data_alvo,coluna_alvo, coluna_chave_posto, nome_cidade):
    print(f"\n--- Analisando o posto: {nome_posto} ---")
    imagem = cv2.imread(caminho_img)  #leitura da imagem
    if imagem is None:    
        print("❌ Erro ao ler a imagem.")
        return df_planilha  
    altura, largura = imagem.shape[:2]
    resultados = modelo.predict(caminho_img, conf=0.15, verbose=False)  #detecção de objetos na imagem
    mapa_classes = modelo.names      #mapa de classes
    inv_mapa = {v: k for k, v in mapa_classes.items()} 
    CLASSE_PRECO = inv_mapa.get('preco', 6)  #variavel para armazenar a classe do preco
    combustiveis_crus, precos = [], []   
    for box in resultados[0].boxes:   #laço para percorrer os boxes em que os resultados foram detectados e separar os combustíveis e preços
        classe, conf_yolo = int(box.cls[0]), float(box.conf[0])   
        x1, y1, x2, y2 = box.xyxy[0].tolist()   
        if ((x2 - x1) * (y2 - y1)) > (altura * largura * 0.40):
            continue    #laço condicional se a area do box for maior que 40% da area da imagem
        dados = {"box": [x1, y1, x2, y2], "conf": conf_yolo, "classe": classe} 
        if classe in mapa_classes and classe != CLASSE_PRECO and mapa_classes[classe] != 'placa':
            dados["nome"] = mapa_classes[classe] #variavel para armazenar o nome do combustivel
            combustiveis_crus.append(dados)
        elif classe == CLASSE_PRECO:   #se o a classe for a classe do preço referido ele armazena o preco
            precos.append(dados)


    print(f"🔎 YOLO: {len(combustiveis_crus)} combustível(is) e {len(precos)} preço(s) detectado(s).")   #fala qual combustivel e preço foi detectado
   
    chave_posto = normalizar_chave(nome_posto)   #normalizando as chaves do posto para comparação de chaves auxiliares e ter a base de quem sera comparado
    chave_cidade = normalizar_chave(nome_cidade)  #normalizando as chaves da cidade 
    
    mascara_posto = (df_planilha[coluna_chave_posto] == chave_posto) #mascara para achar o posto
    mascara_cidade = (df_planilha["_CHAVE_CIDADE"] == chave_cidade) #mascara para achar a cidade
    
    catalogo = df_planilha.loc[mascara_posto & mascara_cidade, col_combustivel].dropna().astype(str).unique().tolist()
    
    
    catalogo_norm = {normalizar_chave(c): c for c in catalogo}
    if not catalogo:
        #Se não achar, mostra no terminal as opções parecidas DENTRO da coluna certa
        postos_parecidos = df_planilha.loc[
            df_planilha[coluna_chave_posto].astype(str).str.contains(chave_posto, na=False),
            coluna_chave_posto
        ].dropna().unique().tolist()
        print(f"❌ Posto '{nome_posto}' não encontrado na planilha do alvo selecionado.")
        if postos_parecidos:  #laço de repetição para quando ter postos parecidos na conjuntura da planilha em que o posto nao foi encontrado
            print(f"   Possíveis cadastros parecidos: {postos_parecidos[:10]}")
        return df_planilha #retorna para a planilha original 

    combustiveis_filtrados = {}   #variavel para armazenar os combustiveis filtrados
    for comb in combustiveis_crus:
        nome_norm = normalizar_chave(comb["nome"])  #normaliza o combustivel
        if nome_norm not in catalogo_norm:
            continue
        if nome_norm not in combustiveis_filtrados or comb["conf"] > combustiveis_filtrados[nome_norm]["conf"]:
            combustiveis_filtrados[nome_norm] = comb   #combustiveis filtrados armazenados na variavel
    
    if not combustiveis_filtrados:
        detectados = [(c.get("nome"), round(c.get("conf", 0), 3)) for c in combustiveis_crus]  #variavel para armazenar os combustiveis detectados
        print("⚠️ O YOLO detectou objetos, mas nenhum combustível bateu com o cadastro desse posto.")
        print(f"   Detectado(s): {detectados}")
        return df_planilha

    
    pares = parear_deteccoes(list(combustiveis_filtrados.values()), precos, largura, altura)  #pareando os combustiveis e preços
    if not pares:
        print("⚠️ O YOLO detectou combustível/preço, mas não conseguiu parear as caixas pela posição na foto.")
        return df_planilha #retorna para a planilha original
    
    data_alvo_norm = normalizar_data(data_alvo)   #normalizando a data
    serie_datas = df_planilha[col_data].apply(normalizar_data) #normalizando as datas
    mascara_data = (serie_datas == data_alvo_norm)  #criando a mascara
    encontrou_algo_para_atualizar = False



    for combustivel_det, preco_det, custo in pares:   #laço para percorrer os pares
        combustivel = combustivel_det["nome"]
        conf_comb = combustivel_det["conf"]
        preco, conf_preco = ler_preco(aumentar_caixa(preco_det["box"], imagem))  #lendo o preco
        if not combustivel or preco is None:
            continue
        if conf_comb < LIMIAR_YOLO or conf_preco < LIMIAR_OCR:
            print(f"⚠️ '{combustivel}' ignorado: OCR {conf_preco:.2f} ou YOLO {conf_comb:.2f} abaixo do mínimo.") #se o combustivel ou o preco for abaixo do limiar, ele ignora
            continue



        chave_combustivel = normalizar_chave(combustivel)  #normalizando o combustivel
        mascara_comb = (df_planilha["_CHAVE_COMB"] == chave_combustivel)    #mascara para pegar o combustivel certo
        indices_alvo = df_planilha.index[mascara_posto & mascara_cidade & mascara_comb & mascara_data]  #pegando os indices
        
        if len(indices_alvo) == 0:
            continue   
        atualizou_algum = False 


        for idx in indices_alvo:   #laço para percorrer os indices 
            preco_antigo_historico = buscar_preco_anterior(  #funcao para buscar o preco anterior
                df_planilha,
                nome_posto,
                combustivel,
                data_alvo_norm,
                coluna_alvo,
                coluna_chave_posto,
                nome_cidade
            )
            precisa_atualizar = False
            print(f"🔎 Histórico: {nome_posto} | {combustivel} | Preço anterior encontrado: {preco_antigo_historico}")  
            if preco_antigo_historico is None:   #limiarem o preco se nenhuma data for encontrada
                if 3.0 <= preco <= 10.00:
                    precisa_atualizar = True  #precisa atualizar se o preco estiver entre 3 e 10
                else:
                    print(f"⚠️ Preço rejeitado sem histórico: {combustivel} R$ {preco:.2f}")
                    precisa_atualizar = False
            else:
                variacao = abs(preco - preco_antigo_historico) / preco_antigo_historico
                if variacao <= LIMITE_VARIACAO_PRECO:   #limitem o preco se a variacao for menor que o limiar
                    precisa_atualizar = True
                else:
                    print(f"⚠️ Preço rejeitado: {combustivel} R$ {preco:.2f} (anterior R$ {preco_antigo_historico:.2f}) variação {variacao:.1%}")
                    precisa_atualizar = False

            if precisa_atualizar:
                df_planilha.loc[idx, coluna_alvo] = preco #Grava dinamicamente o preco
                atualizou_algum = True 
                encontrou_algo_para_atualizar = True   
        if atualizou_algum:
            print(f"✅ ATUALIZADO ({len(indices_alvo)} lugar(es)): {combustivel} -> R$ {preco:.2f}")
    if not encontrou_algo_para_atualizar:
        print("ℹ️ Esta foto terminou sem alteração de preços, os avisos mostram o motivo da rejeição")
    return df_planilha


#==========================================
#5)-O Fluxo de pastas e execução do programa
#==========================================
print("Carregando planilha...")
planilha = pd.read_excel(caminho_planilha)
planilha.columns = planilha.columns.str.strip()  #limpa os espaços das colunas

#para facilitar o carregamento, criarei as chaves auxiliares para posto e combustível o que ajudar a agilidade

planilha["_CHAVE_POSTO_CONC"] = planilha[col_posto].apply(normalizar_chave)
planilha["_CHAVE_POSTO_CLI"] = planilha[col_posto_cliente].apply(normalizar_chave)
planilha["_CHAVE_COMB"] = planilha[col_combustivel].apply(normalizar_chave)
planilha["_CHAVE_CIDADE"] = planilha[col_cidade].apply(normalizar_chave)



#Valida as colunas mínimas que o fluxo precisa.
colunas_obrigatorias = [col_posto, col_posto_cliente, col_combustivel, col_data, col_cidade]
colunas_faltantes = [c for c in colunas_obrigatorias if c not in planilha.columns]
if colunas_faltantes:
    raise KeyError(f"A planilha não possui as colunas obrigatórias: {colunas_faltantes}")

#Garante a existência da coluna PRECO_CONCORRENTE
if col_preco_concorrente not in planilha.columns:
    planilha[col_preco_concorrente] = None

if col_preco_cliente not in planilha.columns:  #Para evitar erros futuros
    planilha[col_preco_cliente] = None

#Normaliza toda a coluna data
#Isso elimina a mistura entre texto DD/MM/AAAA e datas reais do Excel
datas_originais = planilha[col_data].copy()
planilha[col_data] = planilha[col_data].apply(normalizar_data)

qtd_datas_invalidas = int((datas_originais.notna() & planilha[col_data].isna()).sum())
if qtd_datas_invalidas > 0:
    print(f"⚠️ Atenção: {qtd_datas_invalidas} linha(s) possuem DATA que não pôde ser interpretada.")

#Função que pega a data de hoje automaticamente pelo relógio do computador
data_hoje = pd.Timestamp(datetime.datetime.now().date())

#Descobre o último dia válido existente na planilha
datas_validas = planilha[col_data].dropna()
if datas_validas.empty:
    raise ValueError("A coluna DATA não possui nenhuma data válida para usar como modelo.")

ultima_data_cadastrada = datas_validas.max().normalize()

#Se hoje ainda não existe, clona o bloco do último dia cadastrado
mascara_hoje = (planilha[col_data] == data_hoje)
if not mascara_hoje.any():
    print(f"\n📅 Criando novo bloco de registros para o dia de hoje ({data_hoje.strftime('%d/%m/%Y')})...")
    mascara_ultimo_dia = (planilha[col_data] == ultima_data_cadastrada)
    linhas_clonadas = planilha.loc[mascara_ultimo_dia].copy()

    if linhas_clonadas.empty:
        raise ValueError(
            f"Não encontrei linhas do último dia {ultima_data_cadastrada.strftime('%d/%m/%Y')} para clonar."
        )
    linhas_clonadas[col_data] = data_hoje
    planilha = pd.concat([planilha, linhas_clonadas], ignore_index=True)
    
    planilha["_CHAVE_POSTO_CONC"] = planilha[col_posto].apply(normalizar_chave)
    planilha["_CHAVE_POSTO_CLI"] = planilha[col_posto_cliente].apply(normalizar_chave)
    planilha["_CHAVE_COMB"] = planilha[col_combustivel].apply(normalizar_chave)
    planilha["_CHAVE_CIDADE"] = planilha[col_cidade].apply(normalizar_chave)

    print(f"✅ Novo dia criado com {len(linhas_clonadas)} linha(s).")
    print(f"✅ Novo dia criado com {len(linhas_clonadas)} linha(s).")
else:
    print(
        f"\n📅 O dia {data_hoje.strftime('%d/%m/%Y')} já existe na planilha "
        f"com {int(mascara_hoje.sum())} linha(s). Atualizando os registros dele..."
    )
#Mapeia todas as regiões disponíveis
regioes_disponiveis = [
    f for f in os.listdir(pasta_base_fotos)
    if os.path.isdir(os.path.join(pasta_base_fotos, f))
]
if not regioes_disponiveis:
    print(f"❌ Nenhuma pasta de região encontrada em: {pasta_base_fotos}")
    exit()
#Dicionário que conecta a letra digitada com o nome exato da pasta no Windows.
mapa_letras = {
    'A': 'REGIAO_A',
    'B': 'REGIAO_B',
    'C': 'REGIAO_C',
    'D': 'REGIAO_D'
}




#Lógica do terminal Parte 1 - Escolhe o que deseja atualizar
print("\n" + "="*50)
print(" ⛽ SISTEMA DE ATUALIZAÇÃO DE PREÇOS")
print("Selecione o que deseja atualizar:\n")
print("  1) - Preço Postos")
print("  2) - Preço Cliente X")
print("  3) - Preço de Ambos")
print("\n  0) - Sair do programa")
print("="*50)

opcao_alvo = input("Digite a opção desejada: ").strip()

if opcao_alvo == '0':
    print("\nEncerrando o sistema...")
    exit()

#Define quais pastas e colunas serão alimentadas nesta rodada
processamento = []
if opcao_alvo == '1':
    processamento.append((pasta_base_fotos, col_preco_concorrente, "_CHAVE_POSTO_CONC", "Concorrentes"))
elif opcao_alvo == '2':
    processamento.append((pasta_cliente, col_preco_cliente, "_CHAVE_POSTO_CLI", "Clientes"))
elif opcao_alvo == '3':
    processamento.append((pasta_base_fotos, col_preco_concorrente, "_CHAVE_POSTO_CONC", "Concorrentes")) 
    processamento.append((pasta_cliente, col_preco_cliente, "_CHAVE_POSTO_CLI", "Clientes"))
else:  
    print("\n⚠️ Opção inválida! Encerrando.")
    exit()


#Lógica do terminal - Parte referente a escolha das regiões que serão atualizadas
print("\n" + "="*50)
print("Selecione a REGIÃO desejada:\n")
print("  1) - Atualizar TODAS as regiões")
print("  A) - Atualizar REGIAO A")
print("  B) - Atualizar REGIAO B")
print("  C) - Atualizar REGIAO C")
print("  D) - Atualizar REGIAO D")
print("="*50)

opcao_regiao = input("Digite a região: ").strip().upper()

if opcao_regiao == '1':
    regioes_selecionadas = regioes_disponiveis
    print("\n🚀 Preparando para atualizar TODAS as regiões...")
elif opcao_regiao in mapa_letras:
    nome_pasta_escolhida = mapa_letras[opcao_regiao]
    if nome_pasta_escolhida in regioes_disponiveis:   #navega pelo dicionario e verifica se a pasta escolhida existe
        regioes_selecionadas = [nome_pasta_escolhida]
        print(f"\n🚀 Preparando para atualizar apenas: {nome_pasta_escolhida}")
    else:
        print(f"\n⚠️ A pasta '{nome_pasta_escolhida}' não foi encontrada.")
        exit()
else:
    print("\n⚠️ Opção inválida! Encerrando.")     #invalido ou nao existente
    exit()

#Passeia pelos nome_regiao e verifica se existem imagens para serem processadas
for nome_regiao in regioes_selecionadas:
    for caminho_base_alvo, coluna_alvo,coluna_chave_posto, tipo_posto in processamento:
        caminho_regiao = os.path.join(caminho_base_alvo, nome_regiao)
        if os.path.isdir(caminho_regiao):
            print("\n" + "-"*40)
            print(f"📍 INICIANDO REGIÃO: {nome_regiao.upper()} | MODO: {tipo_posto}")
            print("-" * 40)
            for nome_posto in os.listdir(caminho_regiao):
                caminho_posto = os.path.join(caminho_regiao, nome_posto)
                if os.path.isdir(caminho_posto):
                    extensoes = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
                    fotos_posto = []
                    for ext in extensoes:
                        fotos_posto.extend(glob.glob(os.path.join(caminho_posto, ext)))
                    if fotos_posto:
                        ultima_foto = max(fotos_posto, key=os.path.getmtime)
                        #Cria as variáveis de data PRIMEIRO
                        data_foto = datetime.datetime.fromtimestamp(os.path.getmtime(ultima_foto)).date()
                        hoje = datetime.datetime.now().date()
                        #só processa se estritamente a foto for de hoje
                        if data_foto == hoje:
                            print(f"🖼️ Foto nova encontrada: {os.path.basename(ultima_foto)}")
                            #Passa ambas as variáveis dinâmicas para a função corretamente:
                            planilha = processar_foto(ultima_foto, nome_posto, planilha, data_hoje, coluna_alvo, coluna_chave_posto, nome_regiao)
                        else:
                            print(f"⏩ Pulando {nome_posto}: Foto antiga ({data_foto.strftime('%d/%m/%Y')}).")
                    else:
                        print(f"⚠️ Nenhuma foto na pasta do posto: {nome_posto}")
try:
    #Apaga as colunas temporárias para elas não apareçam no Excel
    colunas_ocultas = ["_CHAVE_POSTO_CONC","_CHAVE_COMB", "_CHAVE_POSTO","_CHAVE_CIDADE","_CHAVE_POSTO_CLI", col_preco_cliente]
    planilha_final = planilha.drop(columns=colunas_ocultas, errors='ignore')
    nome_da_aba = 'concorrentes' 
    with pd.ExcelWriter(caminho_planilha, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        planilha_final.to_excel(writer, sheet_name=nome_da_aba, index=False)
    #Formatação da coluna de data para o padrão visual DD/MM/YYYY
    from openpyxl import load_workbook
    wb = load_workbook(caminho_planilha)
    ws = wb[nome_da_aba]
    coluna_data_idx = None
    for coluna in ws[1]:
        if coluna.value == col_data:
            coluna_data_idx = coluna.column
            break
    if coluna_data_idx:
        for linha in range(2, ws.max_row + 1):
            ws.cell(row=linha, column=coluna_data_idx).number_format = "DD/MM/YYYY"
    wb.save(caminho_planilha) #Salvamento
    print("\n" + "="*50)
    print("✅ Fluxo feito! Planilha atualizada")
    print("="*50 + "\n")
except PermissionError:
    print("\n❌ ERRO: O Excel está aberto! feche a aba e tente novamente.")
