#remove as fotos da minha pasta referente a fotos_postos
import os 

Base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#caminho
caminho_retirada = os.path.join(Base_dir,"data","fotos_postos" & "fotos_postos_clientes")

#extensoes que serão removidas
extensoes_fotos = ('.jpg', '.jpeg', '.png')
qtd_deletada = 0
#Percorre todos os arquivos na pasta
for raiz, diretorios, arquivos in os.walk(caminho_retirada):
    for arquivo in arquivos:
        #Verifica se o arquivo termina com uma das extensões de foto
        if arquivo.lower().endswith(extensoes_fotos):
            caminho_completo = os.path.join(raiz, arquivo)
            try:
                os.remove(caminho_completo)
                qtd_deletada += 1
                print(f"Deletado: {arquivo}") 
            except Exception as e:
                print(f"❌ Erro ao deletar {arquivo}: {e}")
print(f"✅ Limpeza concluída! {qtd_deletada} fotos foram removidas, sem nenhuma foto.")

