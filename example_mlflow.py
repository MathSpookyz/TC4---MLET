"""
Exemplo de uso do MLFlow com o sistema de previsão de ações

Este script demonstra como o MLFlow rastreia automaticamente
todas as operações de treinamento e previsão.
"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"


def print_section(title):
    """Imprime cabeçalho de seção"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60 + "\n")


def train_model(ticker, start_date="2020-01-01"):
    """Treina um modelo e rastreia no MLFlow"""
    print_section(f"Treinando modelo para {ticker}")
    
    response = requests.post(
        f"{BASE_URL}/train",
        json={
            "ticker": ticker,
            "start_date": start_date
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Treinamento concluído!")
        print(f"   RMSE: {result.get('rmse', 'N/A')}")
        print(f"   Próxima previsão: R$ {result.get('next_prediction', 'N/A'):.2f}")
        print(f"   Treinado em: {result.get('trained_at', 'N/A')}")
        print(f"\n📊 Veja os detalhes no MLFlow UI: http://localhost:5000")
        return result
    else:
        print(f"❌ Erro: {response.status_code}")
        print(f"   {response.text}")
        return None


def make_prediction(ticker, days=5):
    """Faz uma previsão e rastreia no MLFlow"""
    print_section(f"Fazendo previsão para {ticker} - {days} dias")
    
    response = requests.get(f"{BASE_URL}/predict/{ticker}?days={days}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Previsão concluída!")
        print(f"   Último preço conhecido: R$ {result['last_known_price']:.2f}")
        print(f"   Previsões para {days} dias:")
        
        for i, pred in enumerate(result['predictions'], 1):
            print(f"     Dia {i}: R$ {pred:.2f}")
        
        print(f"\n📊 Veja os detalhes no MLFlow UI: http://localhost:5000")
        return result
    else:
        print(f"❌ Erro: {response.status_code}")
        print(f"   {response.text}")
        return None


def predict_with_custom_data():
    """Faz previsão com dados personalizados e rastreia no MLFlow"""
    print_section("Previsão com dados personalizados")
    
    # Dados de exemplo
    custom_data = {
        "ticker_name": "TESTE_MLFLOW",
        "days": 3,
        "historical_data": [
            {"date": f"2024-01-{str(i).zfill(2)}", "close": 100.0 + i * 0.5, "volume": 1000000 + i * 10000}
            for i in range(1, 46)
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/predict-custom",
        json=custom_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Previsão customizada concluída!")
        print(f"   Nome do ticker: {result['ticker_name']}")
        print(f"   RMSE: {result['rmse']}")
        print(f"   Amostras de treino: {result['training_samples']}")
        print(f"   Último preço conhecido: R$ {result['last_known_price']:.2f}")
        print(f"   Previsões:")
        
        for i, pred in enumerate(result['predictions'], 1):
            print(f"     Dia {i}: R$ {pred:.2f}")
        
        print(f"\n📊 Veja os detalhes no MLFlow UI: http://localhost:5000")
        return result
    else:
        print(f"❌ Erro: {response.status_code}")
        print(f"   {response.text}")
        return None


def main():
    """Função principal que demonstra o uso do MLFlow"""
    
    print("\n🚀 Demonstração do MLFlow com Sistema de Previsão de Ações")
    print("=" * 60)
    print("\nEste script vai:")
    print("1. Treinar um modelo para PETR4.SA")
    print("2. Fazer previsões para 5 dias")
    print("3. Fazer previsão com dados customizados")
    print("\nTodas as operações serão rastreadas automaticamente no MLFlow!")
    print("\n💡 Dica: Abra http://localhost:5000 para ver os experimentos")
    print("\nPressione Enter para continuar...")
    input()
    
    # 1. Treinar modelo
    ticker = "PETR4.SA"
    train_result = train_model(ticker)
    
    if train_result:
        print("\n⏳ Aguardando 2 segundos...")
        time.sleep(2)
        
        # 2. Fazer previsão
        predict_result = make_prediction(ticker, days=5)
        
        if predict_result:
            print("\n⏳ Aguardando 2 segundos...")
            time.sleep(2)
            
            # 3. Previsão customizada
            custom_result = predict_with_custom_data()
    
    # Resumo final
    print_section("Resumo da Demonstração")
    print("✅ Demonstração concluída!")
    print("\n📊 Próximos passos:")
    print("   1. Abra http://localhost:5000 (MLFlow UI)")
    print("   2. Veja o experimento 'stock-price-prediction'")
    print("   3. Compare as métricas (RMSE, previsões)")
    print("   4. Explore os parâmetros e artefatos")
    print("\n📖 Guia completo: MLFLOW_GUIDE.md")
    print("\n🎯 O MLFlow está sempre ativo e rastreando todas as operações!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Operação cancelada pelo usuário")
    except requests.exceptions.ConnectionError:
        print("\n\n❌ Erro: Não foi possível conectar à API")
        print("   Certifique-se de que a API está rodando:")
        print("   python api.py")
    except Exception as e:
        print(f"\n\n❌ Erro inesperado: {e}")
