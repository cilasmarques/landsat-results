#!/usr/bin/env python3
"""
Script para gerar stacked barplot one com três fases:
- Leitura dos dados de entrada (E_READ)
- Cálculo da ET (fases de processamento combinadas)
- Escrita dos dados de saída (E_WRITE)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from plotnine import *
import warnings
warnings.filterwarnings('ignore')

# Estratégias Serial para comparação
CPU_STRATEGIES = [
    'kernels-double-fm-r-sebal', 'kernels-double-fm-r-steep'
]

# Mapeamento para nomes das abordagens
STRATEGY_LABELS = {
    'kernels-double-fm-r-sebal': 'GPU SIMD',
    'kernels-double-fm-r-steep': 'GPU SIMD'
}

def load_macrogroup_data(input_dir):
    """Carrega dados das etapas do diretório summarized_results_grouped"""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {input_path}")
    
    all_data = []
    
    # Listar apenas as estratégias CPU especificadas
    strategy_dirs = [d for d in input_path.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name in CPU_STRATEGIES]
    
    print(f"Carregando dados de {len(strategy_dirs)} estratégias Serial...")
    
    for strategy_dir in strategy_dirs:
        strategy_name = strategy_dir.name
        final_time_file = strategy_dir / 'final-time-grouped.csv'
        
        if not final_time_file.exists():
            print(f"  Aviso: Arquivo não encontrado: {final_time_file}")
            continue
        
        # Carregar dados
        df = pd.read_csv(final_time_file)
        df['strategy_algorithm'] = strategy_name
        all_data.append(df)
        print(f"  Carregado: {strategy_name} ({len(df)} registros)")
    
    if not all_data:
        raise ValueError("Nenhum dado foi carregado")
    
    # Combinar todos os dados
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total de registros carregados: {len(combined_df)}")
    
    return combined_df

def extract_strategy_and_algorithm(df):
    """Extrai estratégia e algoritmo da coluna strategy_algorithm"""
    df = df.copy()
    
    # Mapear strategy_algorithm para strategy e algorithm
    strategy_mapping = {}
    algorithm_mapping = {}
    
    for _, row in df.iterrows():
        strategy_alg = row['strategy_algorithm']
        
        # Determinar algorithm (SEBAL ou STEEP)
        if 'sebal' in strategy_alg.lower():
            algorithm_mapping[strategy_alg] = 'SEBAL'
        elif 'steep' in strategy_alg.lower():
            algorithm_mapping[strategy_alg] = 'STEEP'
        else:
            algorithm_mapping[strategy_alg] = 'UNKNOWN'
        
        # Determinar strategy usando o mapeamento
        strategy_mapping[strategy_alg] = STRATEGY_LABELS.get(strategy_alg, 'OTHER')
    
    df['algorithm'] = df['strategy_algorithm'].map(algorithm_mapping)
    df['strategy'] = df['strategy_algorithm'].map(strategy_mapping)
    df['strategy_plot'] = df['strategy']
    
    return df

def prepare_three_phase_data(macrogroup_data):
    """Prepara dados para as três fases: Leitura, Cálculo e Escrita"""
    # Extrair strategy/algorithm
    df = extract_strategy_and_algorithm(macrogroup_data)
    
    # Fases de processamento para combinar no cálculo da ET
    processing_phases = ['E0_MED_RAD', 'E1_ID_VEG', 'E2_VAR_T', 'E3_MED_RN_G', 'E4_PIX', 'E5_VAR_AR', 'E6_EVAPO']
    
    # Para cada estratégia e algoritmo, combinar as fases de processamento
    new_rows = []
    
    for (strategy, algorithm), group in df.groupby(['strategy_plot', 'algorithm']):
        # Obter dados das fases de processamento
        processing_data = group[group['PHASE'].isin(processing_phases)]
        
        if not processing_data.empty:
            # Somar os tempos das fases de processamento
            total_processing_time = processing_data['TIMESTAMP_median'].sum()
            
            # Criar nova linha para processamento combinado
            processing_row = processing_data.iloc[0].copy()
            processing_row['PHASE'] = 'E_CALCULO'
            processing_row['TIMESTAMP_median'] = total_processing_time
            new_rows.append(processing_row)
    
    # Adicionar as novas linhas de processamento
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    
    # Filtrar apenas as fases que queremos mostrar
    phases_to_keep = ['E_READ', 'E_CALCULO', 'E_WRITE']
    df = df[df['PHASE'].isin(phases_to_keep)]
    
    # Mapeamento das etapas para português
    phase_mapping = {
        'E_READ': 'Leitura dos dados\nde entrada',
        'E_CALCULO': 'Processamento',
        'E_WRITE': 'Escrita dos dados\nde saída'
    }
    
    # Ordem das fases para o plot
    phase_order = [
        'Escrita dos dados\nde saída',
        'Processamento',
        'Leitura dos dados\nde entrada',
    ]
    
    return df, phase_mapping, phase_order

def create_one_stacked_barplot_clean(macrogroup_data, output_dir):
    """Cria stacked barplot one SEM coord_flip e SEM informações dentro das barras"""
    df, phase_mapping, phase_order = prepare_three_phase_data(macrogroup_data)
    
    # Preparar dados para o plot
    plot_data = []    
    for (strategy, algorithm), group in df.groupby(['strategy_plot', 'algorithm'], sort=False):
        for _, row in group.iterrows():
            phase = row['PHASE']
            time_ms = row['TIMESTAMP_median']
            time_sec = time_ms / 1000
            
            plot_data.append({
                'Abordagem': strategy,
                'Algoritmo': algorithm,
                'Fase': phase_mapping[phase],
                'Tempo (s)': time_sec,
                'Tempo (ms)': time_ms
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Converter para Categorical para manter a ordem
    plot_df['Fase'] = pd.Categorical(plot_df['Fase'], categories=phase_order, ordered=True)
    
    # Calcular percentuais para cada estratégia e algoritmo
    for algorithm in plot_df['Algoritmo'].unique():
        for strategy in plot_df['Abordagem'].unique():
            mask = (plot_df['Algoritmo'] == algorithm) & (plot_df['Abordagem'] == strategy)
            if mask.any():
                total_time = plot_df.loc[mask, 'Tempo (s)'].sum()
                plot_df.loc[mask, 'Percentual'] = (plot_df.loc[mask, 'Tempo (s)'] / total_time) * 100
    
    # Adicionar pequena quantidade ao comprimento das barras para acomodar os labels
    plot_df['Tempo (s) Ajustado'] = plot_df['Tempo (s)'] + 0.01
    
    # Criar label combinado: tempo + percentual
    plot_df['Label'] = plot_df['Tempo (s)'].apply(lambda x: f"{x:.3f}") + 's\n(' + plot_df['Percentual'].round(1).astype(str) + '%)'
    
    # A ordem da legenda deve ser INVERSA à ordem das barras
    # Isto garante que a primeira cor na legenda corresponde à barra no topo
    ordem_legenda = list(reversed(phase_order))
    
    # Paletas de cores por algoritmo com as cores específicas fornecidas
    # A ordem das chaves no dicionário de cores define a ordem de plotagem
    # após a ordenação da variável categórica. Para que 'Leitura' fique à esquerda
    # no gráfico horizontal, a sua cor precisa ser a primeira na lista.
    colors_sebal = {
        'Leitura dos dados\nde entrada': '#fc3a3a',     # Azul forte
        'Processamento': '#ff8a8a',                     # Azul médio
        'Escrita dos dados\nde saída': '#bdbfbf',      # Cinza
    }
    
    colors_steep = {
        'Leitura dos dados\nde entrada': '#00bdbd',     # Azul forte
        'Processamento': '#4ECDC4',                     # Azul médio
        'Escrita dos dados\nde saída': '#bdbfbf',      # Cinza
    }
    
    # Também criar versão separada por algoritmo
    for algorithm in ['SEBAL', 'STEEP']:
        alg_data = plot_df[plot_df['Algoritmo'] == algorithm]
        
        if not alg_data.empty:
            # Escolher paleta de cores baseada no algoritmo
            if algorithm == 'SEBAL':
                alg_colors = colors_sebal
            else:  # STEEP
                alg_colors = colors_steep
            
            p_alg = (ggplot(alg_data, aes(x='Abordagem', y='Tempo (s) Ajustado', fill='Fase')) +
                     geom_bar(stat='identity', position='stack', width=0.5) +
                     geom_text(aes(label='Label'), 
                               position=position_stack(vjust=0.5), 
                               size=11, 
                               color='black', 
                               fontweight='bold') +
                     scale_fill_manual(values=alg_colors, name='Fase', limits=ordem_legenda) +
                     scale_y_continuous(breaks=np.arange(0, alg_data['Tempo (s) Ajustado'].max() + 1, 0.2)) +
                     labs(title=f'{algorithm} - Tempo por Fase',
                          x='Algoritmo', 
                          y='Tempo (s)',
                          fill='Fase') +
                     coord_flip() +
                     theme_bw() +
                     theme(axis_text_x=element_text(size=14),
                           axis_text_y=element_text(size=12),
                           axis_title_x=element_text(size=16),
                           axis_title_y=element_text(size=16),
                           plot_title=element_text(size=18, hjust=0.5),
                           legend_text=element_text(size=12),
                           legend_title=element_text(size=14),
                           legend_position='right',
                           figure_size=(15, 3)))  # Altura maior para barras verticais
            
            filename_alg = f"stacked_barplot_serial_{algorithm.lower()}.png"
            p_alg.save(output_dir / filename_alg, dpi=300, bbox_inches='tight')
            print(f"    Salvo: {output_dir / filename_alg}")
            
def main():
    """Função principal"""
    input_dir = Path('summarized-groups')
    output_dir = Path('images/stacked-barplots-one')
    output_dir.mkdir(exist_ok=True)
    
    print("=== GERAÇÃO DE STACKED BARPLOT one SERIAL ===")
    print(f"Diretório de entrada: {input_dir}")
    print(f"Diretório de saída: {output_dir}")
    
    # Carregar dados das etapas
    try:
        macrogroup_data = load_macrogroup_data(input_dir)
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return    
    
    print("\n=== GERANDO STACKED BARPLOT one (BARRAS VERTICAIS LIMPAS) ===")
    create_one_stacked_barplot_clean(macrogroup_data, output_dir)
    
    print("\n=== GERAÇÃO CONCLUÍDA ===")
    print(f"Plots salvos em: {output_dir}")

if __name__ == "__main__":
    main()

