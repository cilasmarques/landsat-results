#!/usr/bin/env python3
"""
Script para gerar plots dos dados de tempo e utilização de recursos das estratégias Landsat.

Este script cria visualizações usando plotnine (ggplot para Python) para:
1. Tempos de processamento das fases principais
2. Utilização de CPU e GPU
3. Comparações entre estratégias e algoritmos
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
from plotnine import *
import warnings
warnings.filterwarnings('ignore')

def set_strategy_order(df):
    """Define a ordem das estratégias nos plots"""
    strategy_order = ['serial', 'kernels-raw', 'kernels-streams-seq', 'kernels-streams-async', 'kernels-pc']
    df['strategy'] = pd.Categorical(df['strategy'], categories=strategy_order, ordered=True)
    return df

def load_time_data(results_dir):
    """Carrega todos os dados de tempo dos arquivos final-time.csv"""
    all_data = []
    
    # Estratégias e algoritmos (ordem específica para os plots)
    strategies = ['serial', 'kernels-raw', 'kernels-streams-seq', 'kernels-streams-async', 'kernels-pc']
    algorithms = ['kernels-0', 'kernels-1']
    
    # Fases principais que queremos analisar
    main_phases = ['P0_READ_INPUT', 'P1_INITIAL_PROD', 'P2_PIXEL_SEL', 
                   'P3_RAH', 'P4_FINAL_PROD', 'P5_COPY_HOST', 'P6_SAVE_PRODS']
    
    for strategy in strategies:
        for algorithm in algorithms:
            file_path = Path(results_dir) / strategy / algorithm / 'final-time.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                # Filtra apenas as fases principais
                df = df[df['PHASE'].isin(main_phases)]
                df['strategy'] = strategy
                df['algorithm'] = algorithm
                all_data.append(df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None

def load_resource_data(results_dir, resource_type):
    """Carrega dados de CPU ou GPU"""
    all_data = []
    
    strategies = ['serial', 'kernels-raw', 'kernels-streams-seq', 'kernels-streams-async', 'kernels-pc']
    algorithms = ['kernels-0', 'kernels-1']
    
    for strategy in strategies:
        for algorithm in algorithms:
            if resource_type == 'gpu' and strategy == 'serial':
                continue  # Serial não tem dados de GPU
                
            file_path = Path(results_dir) / strategy / algorithm / f'{resource_type}-time.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['strategy'] = strategy
                df['algorithm'] = algorithm
                df['resource_type'] = resource_type
                all_data.append(df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None

def create_time_plots(time_data, output_dir):
    """Cria plots de tempo de processamento"""
    print("Gerando plots de tempo de processamento...")
    
    # Aplicar ordem das estratégias
    time_data = set_strategy_order(time_data)
    
    # Plot 1: Comparação de tempos por fase e estratégia (kernels-0)
    df_k0 = time_data[time_data['algorithm'] == 'kernels-0'].copy()
    
    p1 = (ggplot(df_k0, aes(x='PHASE', y='TIMESTAMP_median', fill='strategy')) +
          geom_bar(stat='identity', position='dodge', width=0.7) +
          facet_wrap('~strategy', scales='free_y', ncol=2) +
          labs(title='Tempo de Processamento por Fase - Algoritmo STEEP (kernels-0)',
               x='Fase', y='Tempo (segundos)', fill='Estratégia') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(12, 8),
                legend_position='none'))
    
    p1.save(output_dir / 'tempo_por_fase_kernels0.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'tempo_por_fase_kernels0.png'}")
    
    # Plot 2: Comparação de tempos por fase e estratégia (kernels-1)
    df_k1 = time_data[time_data['algorithm'] == 'kernels-1'].copy()
    
    p2 = (ggplot(df_k1, aes(x='PHASE', y='TIMESTAMP_median', fill='strategy')) +
          geom_bar(stat='identity', position='dodge', width=0.7) +
          facet_wrap('~strategy', scales='free_y', ncol=2) +
          labs(title='Tempo de Processamento por Fase - Algoritmo SEBAL (kernels-1)',
               x='Fase', y='Tempo (segundos)', fill='Estratégia') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(12, 8),
                legend_position='none'))
    
    p2.save(output_dir / 'tempo_por_fase_kernels1.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'tempo_por_fase_kernels1.png'}")
    
    # Plot 3: Comparação entre algoritmos para cada estratégia
    p3 = (ggplot(time_data, aes(x='PHASE', y='TIMESTAMP_median', fill='algorithm')) +
          geom_bar(stat='identity', position='dodge', width=0.7) +
          facet_wrap('~strategy', scales='free_y', ncol=2) +
          labs(title='Comparação de Tempos entre Algoritmos por Estratégia',
               x='Fase', y='Tempo (segundos)', fill='Algoritmo') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(14, 10)))
    
    p3.save(output_dir / 'comparacao_algoritmos.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'comparacao_algoritmos.png'}")
    
    # Plot 4: Heatmap de tempos
    pivot_data = time_data.pivot_table(
        values='TIMESTAMP_median', 
        index='strategy', 
        columns='PHASE', 
        aggfunc='mean'
    )
    
    # Converte para formato longo para o plot
    heatmap_data = pivot_data.reset_index().melt(
        id_vars=['strategy'], 
        var_name='PHASE', 
        value_name='TIMESTAMP_median'
    )
    
    p4 = (ggplot(heatmap_data, aes(x='PHASE', y='strategy', fill='TIMESTAMP_median')) +
          geom_tile() +
          geom_text(aes(label=heatmap_data['TIMESTAMP_median'].round(1)), 
                   size=8, color='white') +
          scale_fill_gradient(low='blue', high='red', name='Tempo (s)') +
          labs(title='Heatmap de Tempos de Processamento',
               x='Fase', y='Estratégia') +
          theme(axis_text_x=element_text(rotation=45, hjust=1),
                figure_size=(12, 6)))
    
    p4.save(output_dir / 'heatmap_tempos.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'heatmap_tempos.png'}")

def create_resource_plots(cpu_data, gpu_data, output_dir):
    """Cria plots de utilização de recursos"""
    print("Gerando plots de utilização de recursos...")
    
    if cpu_data is not None:
        # Aplicar ordem das estratégias
        cpu_data = set_strategy_order(cpu_data)
        
        # Plot CPU: Utilização média por estratégia e algoritmo
        # Plot CPU: Utilização média por estratégia e algoritmo
        # Usa a mediana dos valores máximos para capturar melhor a utilização real
        cpu_summary = cpu_data.groupby(['strategy', 'algorithm']).agg({
            'CPU_USAGE_PERCENTAGE_max': 'median',  # Mediana dos valores máximos
            'MEM_USAGE_PERCENTAGE_max': 'median'   # Mediana dos valores máximos de memória
        }).reset_index()
        
        p5 = (ggplot(cpu_summary, aes(x='strategy', y='CPU_USAGE_PERCENTAGE_max', fill='algorithm')) +
              geom_bar(stat='identity', position='dodge', width=0.7) +
              labs(title='Utilização de CPU por Estratégia e Algoritmo (Mediana dos Máximos)',
                   x='Estratégia', y='CPU (%)', fill='Algoritmo') +
              theme(axis_text_x=element_text(rotation=45, hjust=1),
                    figure_size=(10, 6)))
        
        p5.save(output_dir / 'utilizacao_cpu.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'utilizacao_cpu.png'}")
        
        # Plot CPU Memory
        p6 = (ggplot(cpu_summary, aes(x='strategy', y='MEM_USAGE_PERCENTAGE_max', fill='algorithm')) +
              geom_bar(stat='identity', position='dodge', width=0.7) +
              labs(title='Utilização de Memória RAM por Estratégia e Algoritmo (Mediana dos Máximos)',
                   x='Estratégia', y='Memória (%)', fill='Algoritmo') +
              theme(axis_text_x=element_text(rotation=45, hjust=1),
                    figure_size=(10, 6)))
        
        p6.save(output_dir / 'utilizacao_memoria.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'utilizacao_memoria.png'}")
    
    if gpu_data is not None:
        # Aplicar ordem das estratégias
        gpu_data = set_strategy_order(gpu_data)
        
        # Plot GPU: Utilização média por estratégia e algoritmo
        # Usa a mediana dos valores máximos para capturar melhor a utilização real
        gpu_summary = gpu_data.groupby(['strategy', 'algorithm']).agg({
            'GPU_USAGE_PERCENTAGE_max': 'median',  # Mediana dos valores máximos
            'MEM_USAGE_PERCENTAGE_max': 'median',  # Mediana dos valores máximos de memória
            'POWER_W_median': 'mean',
            'TEMP_C_median': 'mean'
        }).reset_index()
        
        p7 = (ggplot(gpu_summary, aes(x='strategy', y='GPU_USAGE_PERCENTAGE_max', fill='algorithm')) +
              geom_bar(stat='identity', position='dodge', width=0.7) +
              labs(title='Utilização de GPU por Estratégia e Algoritmo (Mediana dos Máximos)',
                   x='Estratégia', y='GPU (%)', fill='Algoritmo') +
              theme(axis_text_x=element_text(rotation=45, hjust=1),
                    figure_size=(10, 6)))
        
        p7.save(output_dir / 'utilizacao_gpu.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'utilizacao_gpu.png'}")
        
        # Plot GPU Memory
        p8 = (ggplot(gpu_summary, aes(x='strategy', y='MEM_USAGE_PERCENTAGE_max', fill='algorithm')) +
              geom_bar(stat='identity', position='dodge', width=0.7) +
              labs(title='Utilização de Memória GPU por Estratégia e Algoritmo (Mediana dos Máximos)',
                   x='Estratégia', y='Memória GPU (%)', fill='Algoritmo') +
              theme(axis_text_x=element_text(rotation=45, hjust=1),
                    figure_size=(10, 6)))
        
        p8.save(output_dir / 'utilizacao_memoria_gpu.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'utilizacao_memoria_gpu.png'}")
        
        # Plot GPU Power
        p9 = (ggplot(gpu_summary, aes(x='strategy', y='POWER_W_median', fill='algorithm')) +
              geom_bar(stat='identity', position='dodge', width=0.7) +
              labs(title='Consumo Médio de Energia GPU por Estratégia e Algoritmo',
                   x='Estratégia', y='Potência (W)', fill='Algoritmo') +
              theme(axis_text_x=element_text(rotation=45, hjust=1),
                    figure_size=(10, 6)))
        
        p9.save(output_dir / 'consumo_energia_gpu.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'consumo_energia_gpu.png'}")
        
        # Plot GPU Temperature
        p10 = (ggplot(gpu_summary, aes(x='strategy', y='TEMP_C_median', fill='algorithm')) +
               geom_bar(stat='identity', position='dodge', width=0.7) +
               labs(title='Temperatura Média GPU por Estratégia e Algoritmo',
                    x='Estratégia', y='Temperatura (°C)', fill='Algoritmo') +
               theme(axis_text_x=element_text(rotation=45, hjust=1),
                     figure_size=(10, 6)))
        
        p10.save(output_dir / 'temperatura_gpu.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'temperatura_gpu.png'}")

def create_comparison_plots(time_data, output_dir):
    """Cria plots de comparação de tempos por fase e estratégia"""
    print("Gerando plots de comparação...")
    
    # Aplicar ordem das estratégias
    time_data = set_strategy_order(time_data)
    
    # Calcular a mediana do tempo para cada fase e estratégia
    mean_phases = time_data.groupby(['strategy', 'algorithm', 'PHASE']).agg({
        'TIMESTAMP_median': 'median'
    }).reset_index()
    
    # Converter para segundos (assumindo que está em milissegundos)
    mean_phases['time_seconds'] = mean_phases['TIMESTAMP_median'] / 1000
    
    # Definir ordem das fases
    phase_order = ['P0_READ_INPUT', 'P1_INITIAL_PROD', 'P2_PIXEL_SEL', 'P3_RAH', 'P4_FINAL_PROD', 'P6_SAVE_PRODS']
    
    # Filtrar apenas as fases que queremos mostrar
    mean_phases = mean_phases[mean_phases['PHASE'].isin(phase_order)]
    
    # Criar diferentes visualizações
    # 1. Todas as abordagens
    df1 = mean_phases.copy()
    df1['version'] = '01_Todas_abordagens'
    
    # 2. Apenas abordagens GPU (excluir serial)
    df2 = mean_phases[mean_phases['strategy'] != 'serial'].copy()
    df2['version'] = '02_Apenas_GPU'
    
    # 3. GPU sem leitura serial (excluir P0_READ_INPUT e serial)
    df3 = mean_phases[(mean_phases['strategy'] != 'serial') & 
                      (mean_phases['PHASE'] != 'P0_READ_INPUT')].copy()
    df3['version'] = '03_GPU_sem_leitura'
    
    # Combinar os dataframes
    combined_df = pd.concat([df1, df2, df3], ignore_index=True)
    
    # Criar uma coluna combinada para o facet_wrap
    combined_df['algorithm_version'] = combined_df['algorithm'] + ' - ' + combined_df['version']
    
    # Definir cores para as fases
    phase_colors = {
        'P0_READ_INPUT': '#FF8080',
        'P1_INITIAL_PROD': '#FFB347', 
        'P2_PIXEL_SEL': '#FFFF66',
        'P3_RAH': '#90EE90',
        'P4_FINAL_PROD': '#8080FF',
        'P6_SAVE_PRODS': '#9B7CB9'
    }
    
    # Criar o plot
    p_comparison = (ggplot(combined_df, aes(x='strategy', y='time_seconds', fill='PHASE')) +
                   geom_bar(stat='identity', position='stack', width=0.8) +
                   facet_wrap('algorithm_version', scales='free_y', ncol=3) +
                   scale_fill_manual(values=phase_colors) +
                   labs(title='Comparação de Tempos de Execução por Fase e Estratégia',
                        y='Tempo (segundos)', x='Estratégia', fill='Fase') +
                   theme_bw() +
                   theme(legend_position='bottom',
                         figure_size=(15, 10),
                         axis_text_x=element_text(rotation=45, hjust=1)))
    
    p_comparison.save(output_dir / 'comparacao_tempos_fases.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'comparacao_tempos_fases.png'}")

def create_summary_plots(time_data, cpu_data, gpu_data, output_dir):
    """Cria plots de resumo geral"""
    print("Gerando plots de resumo...")
    
    # Aplicar ordem das estratégias
    time_data = set_strategy_order(time_data)
    
    # Calcula tempo total por estratégia e algoritmo
    total_time = time_data.groupby(['strategy', 'algorithm'])['TIMESTAMP_median'].sum().reset_index()
    total_time.columns = ['strategy', 'algorithm', 'total_time']
    
    p11 = (ggplot(total_time, aes(x='strategy', y='total_time', fill='algorithm')) +
           geom_bar(stat='identity', position='dodge', width=0.7) +
           labs(title='Tempo Total de Processamento por Estratégia e Algoritmo',
                x='Estratégia', y='Tempo Total (segundos)', fill='Algoritmo') +
           theme(axis_text_x=element_text(rotation=45, hjust=1),
                 figure_size=(10, 6)))
    
    p11.save(output_dir / 'tempo_total.png', dpi=300, bbox_inches='tight')
    print(f"  Salvo: {output_dir / 'tempo_total.png'}")
    
    # Comparação de eficiência (tempo vs recursos)
    if cpu_data is not None and gpu_data is not None:
        # Combina dados de tempo e recursos
        efficiency_data = total_time.copy()
        
        # Adiciona dados de CPU (usando mediana dos máximos)
        cpu_avg = cpu_data.groupby(['strategy', 'algorithm']).agg({
            'CPU_USAGE_PERCENTAGE_max': 'median'
        }).reset_index()
        efficiency_data = efficiency_data.merge(cpu_avg, on=['strategy', 'algorithm'])
        
        # Adiciona dados de GPU (usando mediana dos máximos)
        gpu_avg = gpu_data.groupby(['strategy', 'algorithm']).agg({
            'GPU_USAGE_PERCENTAGE_max': 'median'
        }).reset_index()
        efficiency_data = efficiency_data.merge(gpu_avg, on=['strategy', 'algorithm'])
        
        # Calcula eficiência (tempo / utilização de recursos)
        efficiency_data['cpu_efficiency'] = efficiency_data['total_time'] / efficiency_data['CPU_USAGE_PERCENTAGE_max']
        efficiency_data['gpu_efficiency'] = efficiency_data['total_time'] / efficiency_data['GPU_USAGE_PERCENTAGE_max']
        
        p12 = (ggplot(efficiency_data, aes(x='strategy', y='cpu_efficiency', fill='algorithm')) +
               geom_bar(stat='identity', position='dodge', width=0.7) +
               labs(title='Eficiência CPU (Tempo/Utilização) por Estratégia e Algoritmo',
                    x='Estratégia', y='Eficiência CPU', fill='Algoritmo') +
               theme(axis_text_x=element_text(rotation=45, hjust=1),
                     figure_size=(10, 6)))
        
        p12.save(output_dir / 'eficiencia_cpu.png', dpi=300, bbox_inches='tight')
        print(f"  Salvo: {output_dir / 'eficiencia_cpu.png'}")

def main():
    parser = argparse.ArgumentParser(description='Gera plots dos dados de tempo e recursos Landsat')
    parser.add_argument('--input', '-i', type=str, default='results',
                       help='Caminho para o diretório results (padrão: results)')
    parser.add_argument('--output', '-o', type=str, default='plots',
                       help='Caminho para salvar os plots (padrão: plots)')
    
    args = parser.parse_args()
    
    # Configuração dos caminhos
    results_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Verifica se o diretório de entrada existe
    if not results_dir.exists():
        print(f"Erro: Diretório de entrada não encontrado: {results_dir}")
        sys.exit(1)
    
    # Cria o diretório de saída se não existir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== GERANDO PLOTS DOS DADOS LANDSAT ===")
    print(f"Diretório de entrada: {results_dir}")
    print(f"Diretório de saída: {output_dir}")
    
    # Carrega os dados
    print("\nCarregando dados...")
    time_data = load_time_data(results_dir)
    if time_data is not None:
        print(f"  Dados de tempo carregados: {len(time_data)} registros")
    else:
        print("  Erro: Nenhum dado de tempo encontrado")
        sys.exit(1)
    
    cpu_data = load_resource_data(results_dir, 'cpu')
    if cpu_data is not None:
        print(f"  Dados de CPU carregados: {len(cpu_data)} registros")
    
    gpu_data = load_resource_data(results_dir, 'gpu')
    if gpu_data is not None:
        print(f"  Dados de GPU carregados: {len(gpu_data)} registros")
    
    # Gera os plots
    print("\n=== GERANDO PLOTS ===")
    
    # Plots de tempo
    create_time_plots(time_data, output_dir)
    
    # Plots de recursos
    create_resource_plots(cpu_data, gpu_data, output_dir)

    # Plots de comparação
    create_comparison_plots(time_data, output_dir)
    
    # Plots de resumo
    create_summary_plots(time_data, cpu_data, gpu_data, output_dir)
    
    print(f"\n=== PLOTS GERADOS COM SUCESSO ===")
    print(f"Todos os plots foram salvos em: {output_dir}")
    
    # Lista os arquivos gerados
    plot_files = list(output_dir.glob('*.png'))
    print(f"\nArquivos gerados ({len(plot_files)}):")
    for file in sorted(plot_files):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main() 