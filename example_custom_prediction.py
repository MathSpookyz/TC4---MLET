"""
Exemplo de uso do endpoint /predict-custom
Demonstra como enviar dados históricos personalizados para treinamento e previsão
"""

import requests
import json
from datetime import datetime, timedelta

def generate_sample_data(start_date: str, num_days: int = 50):
    """
    Gera dados de exemplo para teste
    Simula uma ação com tendência de alta e alguma volatilidade
    """
    import random
    
    data = []
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    base_price = 100.0
    base_volume = 1000000
    
    for i in range(num_days):
        price_change = random.uniform(-2, 3)
        base_price += price_change
        base_price = max(base_price, 50)
        
        volume_change = random.uniform(-0.1, 0.15)
        base_volume = int(base_volume * (1 + volume_change))
        base_volume = max(base_volume, 500000)
        
        data.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "close": round(base_price, 2),
            "volume": base_volume
        })
        
        current_date += timedelta(days=1)
    
    return data


def test_custom_prediction():
    """
    Testa o endpoint /predict-custom com dados gerados
    """
    print("=" * 60)
    print("TESTE DO ENDPOINT /predict-custom")
    print("=" * 60)
    
    print("\n1. Gerando dados históricos de exemplo...")
    historical_data = generate_sample_data("2024-01-01", num_days=60)
    print(f"   ✓ {len(historical_data)} pontos gerados")
    print(f"   - Primeiro ponto: {historical_data[0]}")
    print(f"   - Último ponto: {historical_data[-1]}")
    
    payload = {
        "ticker_name": "EXEMPLO_TESTE",
        "days": 7,
        "historical_data": historical_data
    }
    
    print("\n2. Enviando requisição para a API...")
    print(f"   - URL: http://localhost:8000/predict-custom")
    print(f"   - Dias para prever: {payload['days']}")
    print(f"   - Pontos históricos: {len(historical_data)}")
    
    try:
        response = requests.post(
            "http://localhost:8000/predict-custom",
            json=payload,
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n3. ✓ Previsão realizada com sucesso!")
            print("=" * 60)
            print(f"\nTicker: {result['ticker_name']}")
            print(f"Último preço conhecido: R$ {result['last_known_price']:.2f}")
            print(f"RMSE do modelo: {result['rmse']:.2f}")
            print(f"Amostras de treinamento: {result['training_samples']}")
            print(f"\nPrevisões para os próximos {result['days']} dias:")
            
            for i, pred in enumerate(result['predictions'], 1):
                print(f"  Dia {i}: R$ {pred:.2f}")
            
            print(f"\nMensagem: {result['message']}")
            print("=" * 60)
            
            with open("custom_prediction_result.json", "w") as f:
                json.dump(result, f, indent=2)
            print("\n✓ Resultado salvo em: custom_prediction_result.json")
            
            return True
            
        else:
            print(f"\n✗ Erro na requisição: {response.status_code}")
            print(f"Detalhes: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n✗ Erro: Não foi possível conectar à API")
        print("Certifique-se de que a API está rodando em http://localhost:8000")
        print("Execute: python api.py")
        return False
    except requests.exceptions.Timeout:
        print("\n✗ Erro: Timeout na requisição")
        print("O treinamento pode estar demorando muito")
        return False
    except Exception as e:
        print(f"\n✗ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_pattern():
    """
    Testa com padrão de dados mais realista
    """
    print("\n" + "=" * 60)
    print("TESTE COM PADRÃO REALISTA")
    print("=" * 60)
    
    historical_data = []
    base_price = 50.0
    
    patterns = [
        (1.2, 1.1, 1.3, 1.0, 1.4),
        (0.5, -0.3, 0.4, -0.2, 0.6),
        (2.0, 1.8, 2.2, 1.5, 2.5),
        (-1.0, -0.8, -1.2, -0.5, -0.7),
        (1.5, 1.3, 1.7, 1.2, 1.6),
        (0.8, 1.0, 0.9, 1.1, 1.2),
        (0.3, 0.2, 0.4, 0.1, 0.5),
        (1.8, 2.0, 1.9, 2.1, 2.2)
    ]
    
    current_date = datetime(2024, 1, 1)
    
    for week, changes in enumerate(patterns, 1):
        for day, change in enumerate(changes, 1):
            base_price += change
            volume = 1000000 + (week * 50000) + (day * 10000)
            
            historical_data.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "close": round(base_price, 2),
                "volume": volume
            })
            
            current_date += timedelta(days=1)
    
    print(f"\n✓ Gerados {len(historical_data)} pontos com padrão realista")
    print(f"  - Preço inicial: R$ {historical_data[0]['close']:.2f}")
    print(f"  - Preço final: R$ {historical_data[-1]['close']:.2f}")
    print(f"  - Variação total: {((historical_data[-1]['close'] / historical_data[0]['close']) - 1) * 100:.2f}%")
    
    payload = {
        "ticker_name": "ACAO_REALISTA",
        "days": 5,
        "historical_data": historical_data
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/predict-custom",
            json=payload,
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✓ Previsão com padrão realista concluída!")
            print(f"  - RMSE: {result['rmse']:.2f}")
            print(f"  - Previsões: {[f'R$ {p:.2f}' for p in result['predictions']]}")
            return True
        else:
            print(f"✗ Erro: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Erro: {e}")
        return False


def main():
    """
    Executa os testes
    """
    print("\n" + "=" * 60)
    print("EXEMPLOS DE USO DO ENDPOINT /predict-custom")
    print("=" * 60)
    print("\nEste script demonstra como usar o endpoint /predict-custom")
    print("que permite treinar e prever com dados históricos personalizados.\n")
    
    print("Antes de executar, certifique-se de que a API está rodando:")
    print("  python api.py\n")
    
    input("Pressione ENTER para continuar...")
    
    success1 = test_custom_prediction()
    
    success2 = test_with_real_pattern()
    
    print("\n" + "=" * 60)
    print("RESUMO DOS TESTES")
    print("=" * 60)
    print(f"Teste 1 (Dados aleatórios): {'✓ PASSOU' if success1 else '✗ FALHOU'}")
    print(f"Teste 2 (Padrão realista): {'✓ PASSOU' if success2 else '✗ FALHOU'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
