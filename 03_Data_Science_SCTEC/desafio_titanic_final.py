#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DESAFIO TITANIC — SCTEC/SENAI LAB365
Aluno: Helyton Renato Gonçalves Ronchi
Curso: Introdução ao Data Science (IP 20h A)
Data: Março/2026
================================================================================
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Compatibilidade com ambientes sem interface gráfica
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ==============================================================================
# CONFIGURAÇÕES INICIAIS
# ==============================================================================

# Caminho dinâmico para o CSV (funciona em qualquer pasta)
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, "titanic.csv")

# Verificação de segurança
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Arquivo titanic.csv não encontrado em: {csv_path}")

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

print("="*70)
print("DESAFIO TITANIC — ANÁLISE EXPLORATÓRIA DE DADOS")
print("="*70)
print(f"Aluno: Helyton Renato Gonçalves Ronchi")
print(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")

# ==============================================================================
# OBJETIVO DO PROJETO
# ==============================================================================

print("OBJETIVO DO PROJETO:")
print("""
Realizar uma análise exploratória do dataset Titanic, aplicando conceitos 
fundamentais de Data Science como limpeza de dados, tratamento de valores 
ausentes, criação de variáveis auxiliares e visualização estatística.
""")

# ==============================================================================
# 1. IMPORTAÇÃO E COMPREENSÃO DOS DADOS
# ==============================================================================

print("\n" + "="*70)
print("ETAPA 1: IMPORTAÇÃO E COMPREENSÃO DOS DADOS")
print("-" * 70)

df = pd.read_csv(csv_path)
print(f"✓ Dataset carregado: {df.shape[0]} linhas × {df.shape[1]} colunas\n")

print("Primeiras 5 linhas:")
print(df.head())

print("\nInformações das colunas:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe())

# ==============================================================================
# 2. TRATAMENTO DE DADOS (LIMPEZA)
# ==============================================================================

print("\n" + "="*70)
print("ETAPA 2: TRATAMENTO DE DADOS")
print("-" * 70)

# 2.1 Identificar valores nulos
print("\nValores nulos por coluna:")
nulos = df.isnull().sum()
print(nulos[nulos > 0])

# 2.2 Tratamento da coluna 'Age' (Idade)
# Estratégia: mediana por classe social (justificativa: idades variavam por classe)
print("\n✓ Tratamento de 'Age':")
idade_por_classe = df.groupby('Pclass')['Age'].median()
for classe in [1, 2, 3]:
    print(f"  Classe {classe}: mediana {idade_por_classe[classe]:.1f} anos")
    df.loc[(df['Age'].isnull()) & (df['Pclass'] == classe), 'Age'] = idade_por_classe[classe]
print(f"  Nulos restantes: {df['Age'].isnull().sum()}")

# 2.3 Tratamento da coluna 'Embarked' (Porto)
print("\n✓ Tratamento de 'Embarked':")
moda_embarked = df['Embarked'].mode()[0]
print(f"  Moda: {moda_embarked} (Southampton)")
df['Embarked'] = df['Embarked'].fillna(moda_embarked)
print(f"  Nulos restantes: {df['Embarked'].isnull().sum()}")

# 2.4 Tratamento da coluna 'Cabin' (Cabine)
# Estratégia: transformar em flag binária (ter ou não cabine)
print("\n✓ Tratamento de 'Cabin':")
df['Has_Cabin'] = df['Cabin'].notna().astype(int)
df = df.drop(columns=['Cabin'])
print(f"  Criada flag 'Has_Cabin' (0=sem cabine, 1=com cabine)")

# 2.5 Verificação de duplicatas
print(f"\n✓ Duplicatas: {df.duplicated().sum()}")

# 2.6 Ajuste de tipos
df['Survived'] = df['Survived'].astype(int)
df['Pclass'] = df['Pclass'].astype(int)
print("✓ Tipos ajustados")

print("\nDataset após tratamento:")
print(df.info())

# ==============================================================================
# 3. ANÁLISE EXPLORATÓRIA
# ==============================================================================

print("\n" + "="*70)
print("ETAPA 3: ANÁLISE EXPLORATÓRIA")
print("-" * 70)

# 3.1 Taxa geral
taxa_geral = df['Survived'].mean() * 100
print(f"\n✓ Taxa geral de sobrevivência: {taxa_geral:.2f}% ({df['Survived'].sum()} de {len(df)})")

# 3.2 Por classe social
print("\n✓ Sobrevivência por Classe Social:")
sobrev_classe = df.groupby('Pclass')['Survived'].agg(['mean', 'count'])
sobrev_classe_percent = (sobrev_classe['mean'] * 100).round(1)
for classe in [1, 2, 3]:
    print(f"  Classe {classe}: {sobrev_classe_percent[classe]:.1f}% ({sobrev_classe.loc[classe, 'count']} passageiros)")

# 3.3 Por sexo
print("\n✓ Sobrevivência por Sexo:")
sobrev_sexo = df.groupby('Sex')['Survived'].agg(['mean', 'count'])
sobrev_sexo_percent = (sobrev_sexo['mean'] * 100).round(1)
print(f"  Mulheres: {sobrev_sexo_percent['female']:.1f}% ({sobrev_sexo.loc['female', 'count']} passageiros)")
print(f"  Homens: {sobrev_sexo_percent['male']:.1f}% ({sobrev_sexo.loc['male', 'count']} passageiros)")

# 3.4 Por porto
print("\n✓ Sobrevivência por Porto de Embarque:")
sobrev_porto = df.groupby('Embarked')['Survived'].agg(['mean', 'count'])
sobrev_porto_percent = (sobrev_porto['mean'] * 100).round(1)
for porto in ['C', 'Q', 'S']:
    if porto in sobrev_porto_percent.index:
        nome_porto = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}[porto]
        print(f"  {nome_porto}: {sobrev_porto_percent[porto]:.1f}% ({sobrev_porto.loc[porto, 'count']} passageiros)")

# 3.5 Análise de crianças
criancas = df[df['Age'] < 12]
taxa_criancas = criancas['Survived'].mean() * 100
print(f"\n✓ Crianças (<12 anos): {len(criancas)} passageiros, taxa {taxa_criancas:.1f}%")

# 3.6 Análise de idosos
idosos = df[df['Age'] > 60]
taxa_idosos = idosos['Survived'].mean() * 100
print(f"✓ Idosos (>60 anos): {len(idosos)} passageiros, taxa {taxa_idosos:.1f}%")

# 3.7 Por cabine
taxa_com_cabine = df[df['Has_Cabin'] == 1]['Survived'].mean() * 100
taxa_sem_cabine = df[df['Has_Cabin'] == 0]['Survived'].mean() * 100
print(f"\n✓ Com cabine: {taxa_com_cabine:.1f}%")
print(f"✓ Sem cabine: {taxa_sem_cabine:.1f}%")

# 3.8 Feature engineering simples
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)

sozinho = df[df['Is_Alone'] == 1]['Survived'].mean() * 100
acompanhado = df[df['Is_Alone'] == 0]['Survived'].mean() * 100
print(f"\n✓ Sozinho: {sozinho:.1f}%")
print(f"✓ Acompanhado: {acompanhado:.1f}%")

# 3.9 Mais velhos sobreviventes
print("\n✓ 5 passageiros mais velhos que sobreviveram:")
mais_velhos = df[df['Survived'] == 1].sort_values('Age', ascending=False).head(5)
print(mais_velhos[['Name', 'Age', 'Sex', 'Pclass']])

# ==============================================================================
# 4. VISUALIZAÇÕES
# ==============================================================================

print("\n" + "="*70)
print("ETAPA 4: VISUALIZAÇÕES")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análise Titanic — Desafio SCTEC/SENAI', fontsize=14, fontweight='bold')

# Gráfico 1: Sobrevivência por Classe e Sexo
ax1 = axes[0, 0]
sns.barplot(data=df, x='Pclass', y='Survived', hue='Sex', ax=ax1,
            palette=['#e74c3c', '#2ecc71'])
ax1.set_title('Sobrevivência por Classe e Sexo')
ax1.set_xlabel('Classe')
ax1.set_ylabel('Taxa de Sobrevivência')
ax1.legend(title='Sexo')

# Gráfico 2: Distribuição de Idade
ax2 = axes[0, 1]
sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True, ax=ax2,
             palette=['#e74c3c', '#2ecc71'])
ax2.set_title('Distribuição de Idade por Sobrevivência')
ax2.set_xlabel('Idade (anos)')
ax2.set_ylabel('Frequência')
handles, _ = ax2.get_legend_handles_labels()
ax2.legend(handles, ['Não Sobreviveu', 'Sobreviveu'])

# Gráfico 3: Tarifa por Classe
ax3 = axes[1, 0]
sns.boxplot(data=df, x='Pclass', y='Fare', hue='Survived', ax=ax3,
            palette=['#e74c3c', '#2ecc71'])
ax3.set_title('Tarifa por Classe e Sobrevivência')
ax3.set_xlabel('Classe')
ax3.set_ylabel('Tarifa (£)')
ax3.legend(labels=['Não Sobreviveu', 'Sobreviveu'])
ax3.set_ylim(0, 300)

# Gráfico 4: Sobrevivência por Porto
ax4 = axes[1, 1]
sns.countplot(data=df, x='Embarked', hue='Survived', ax=ax4,
              palette=['#e74c3c', '#2ecc71'])
ax4.set_title('Sobrevivência por Porto de Embarque')
ax4.set_xlabel('Porto (C=Cherbourg, Q=Queenstown, S=Southampton)')
ax4.set_ylabel('Quantidade de Passageiros')
ax4.legend(labels=['Não Sobreviveu', 'Sobreviveu'])

plt.tight_layout()
plt.savefig('graficos_titanic.png', dpi=150, bbox_inches='tight')
print("✓ Gráficos salvos em: graficos_titanic.png")

# ==============================================================================
# 5. SALVAR DADOS TRATADOS
# ==============================================================================

df.to_csv('titanic_tratado.csv', index=False, encoding='utf-8')
print(f"\n✓ Dados tratados salvos em: titanic_tratado.csv")

# ==============================================================================
# 6. LIMITAÇÕES DO ESTUDO
# ==============================================================================

print("\n" + "="*70)
print("LIMITAÇÕES DO ESTUDO")
print("-" * 70)
print("""
• O dataset não possui informação detalhada sobre prioridade real de resgate
  (ordem de acesso aos botes, por exemplo).

• A variável 'Cabin' possui aproximadamente 77% de valores ausentes, o que
  limitou sua análise a uma flag binária.

• A imputação de idade por mediana da classe, embora estatisticamente 
  justificável, pode introduzir viés se a distribuição real for complexa.

• A análise é exploratória e não estabelece relações de causalidade —
  apenas correlações observadas nos dados disponíveis.
""")

# ==============================================================================
# 7. REFLEXÃO DO ALUNO
# ==============================================================================

print("\n" + "="*70)
print("REFLEXÃO DO ALUNO")
print("-" * 70)
print("""
Este projeto representou meu primeiro contato estruturado com análise 
exploratória de dados reais. Durante o desenvolvimento, pude compreender 
a importância da limpeza, tratamento e organização dos dados antes de 
qualquer interpretação.

Percebi que pequenas decisões (como a escolha da mediana para imputação 
de idades, ou a transformação de 'Cabin' em flag) impactam diretamente 
na qualidade da análise e nos insights gerados.

Este exercício reforçou meu interesse pela área de Dados e me motivou 
a aprofundar conhecimentos em estatística, modelagem preditiva e boas 
práticas de programação.

Projeto desenvolvido como parte do curso introdutório de Data Science, 
com aprofundamentos e melhorias realizadas de forma autônoma.
""")

# ==============================================================================
# 8. PRÓXIMOS PASSOS FUTUROS
# ==============================================================================

print("\n" + "="*70)
print("PRÓXIMOS PASSOS FUTUROS")
print("-" * 70)
print("""
• Aplicar modelo de Machine Learning (ex: Regressão Logística) para 
  previsão de sobrevivência com base nas variáveis disponíveis.

• Testar outras estratégias de imputação para dados faltantes 
  (ex: regressão, KNN) e comparar resultados.

• Avaliar o impacto de novas features (tamanho da família, títulos, 
  etc.) na qualidade da análise e em possíveis modelos preditivos.

• Explorar validação cruzada para garantir robustez dos resultados.
""")

# ==============================================================================
# 9. INSIGHTS E CONCLUSÕES
# ==============================================================================

print("\n" + "="*70)
print("INSIGHTS E CONCLUSÕES")
print("-" * 70)

print(f"""
1. TAXA GERAL: {taxa_geral:.1f}% de sobrevivência

2. CLASSE SOCIAL:
   • 1ª Classe: {sobrev_classe_percent[1]:.1f}%
   • 2ª Classe: {sobrev_classe_percent[2]:.1f}%
   • 3ª Classe: {sobrev_classe_percent[3]:.1f}%

3. GÊNERO:
   • Mulheres: {sobrev_sexo_percent['female']:.1f}%
   • Homens: {sobrev_sexo_percent['male']:.1f}%

4. IDADE:
   • Crianças (<12 anos): {taxa_criancas:.1f}%
   • Idosos (>60 anos): {taxa_idosos:.1f}%

5. CABINE:
   • Com cabine: {taxa_com_cabine:.1f}%
   • Sem cabine: {taxa_sem_cabine:.1f}%

6. FAMÍLIA:
   • Sozinho: {sozinho:.1f}%
   • Acompanhado: {acompanhado:.1f}%

7. PORTO:
   • Cherbourg (C): {sobrev_porto_percent['C']:.1f}%
   • Queenstown (Q): {sobrev_porto_percent['Q']:.1f}%
   • Southampton (S): {sobrev_porto_percent['S']:.1f}%

CONCLUSÃO:
Os resultados indicam forte correlação entre classe social, gênero e 
sobrevivência. Mulheres e passageiros da 1ª classe apresentaram maior 
taxa de sobrevivência, reforçando evidências históricas do protocolo 
"mulheres e crianças primeiro" e da desigualdade estrutural no acesso 
aos botes salva-vidas.
""")

print("="*70)
print(f"Análise concluída em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("="*70)
