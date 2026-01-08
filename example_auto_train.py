"""
Exemplo de uso da funcionalidade de treinamento automático
Demonstra como o sistema treina automaticamente quando modelo não existe
"""

import requests
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def test_auto_training():
    """
    Testa o treinamento automático fazendo uma previsão para um ticker
    que provavelmente não tem modelo treinado
    """
    print("=" * 70)
    print("TESTE DE TREINAMENTO AUTOMÁTICO")
    print("=" * 70)
    
    ticker = "BBDC4.SA"
    days = 5
    
    print(f"\n1. Tentando fazer previsão para {ticker}")
    print(f"   - Dias: {days}")
    print(f"   - URL: http://localhost:8000/predict/{ticker}")
    print("\nSe o modelo não existir, o sistema irá:")
    print("   a) Detectar que o modelo não existe")
    print("   b) Buscar dados históricos do Yahoo Finance")
    print("   c) Treinar o modelo automaticamente")
    print("   d) Fazer a previsão solicitada")
    print("\nIsso pode levar alguns minutos...")
    print("=" * 70)
    
    try:
        start_time = time.time()
        
        response = requests.get(
            f"http://localhost:8000/predict/{ticker}",
            params={"days": days},
            timeout=600
        )
        
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✓ SUCESSO!")
            print("=" * 70)
            print(f"\nTempo total: {elapsed_time:.1f} segundos")
            print(f"\nTicker: {result['ticker']}")
            print(f"Último preço conhecido: R$ {result['last_known_price']:.2f}")
            print(f"\nPrevisões para os próximos {result['days']} dias:")
            
            for i, pred in enumerate(result['predictions'], 1):
                print(f"  Dia {i}: R$ {pred:.2f}")
            
            print("\n" + "=" * 70)
            print("OBSERVAÇÕES:")
            print("- O modelo foi treinado automaticamente")
            print("- Próximas previsões para este ticker serão mais rápidas")
            print(f"- Modelo salvo em: export/lstm_model_{ticker}.pth")
            print("=" * 70)
            
            return True
            
        else:
            print(f"\n✗ Erro na requisição: {response.status_code}")
            print(f"Detalhes: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n✗ Erro: Não foi possível conectar à API")
        print("Certifique-se de que a API está rodando:")
        print("  python api.py")
        return False
    except requests.exceptions.Timeout:
        print("\n✗ Erro: Timeout na requisição")
        print("O treinamento pode estar demorando muito")
        print("Tente aumentar o timeout ou verificar os logs da API")
        return False
    except Exception as e:
        print(f"\n✗ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cached_prediction():
    """
    Testa uma segunda previsão para o mesmo ticker
    (deve ser mais rápida pois o modelo já está treinado)
    """
    print("\n\n" + "=" * 70)
    print("TESTE DE PREVISÃO COM MODELO EM CACHE")
    print("=" * 70)
    
    ticker = "BBDC4.SA"
    days = 7
    
    print(f"\nFazendo segunda previsão para {ticker}")
    print("Desta vez deve ser muito mais rápido!")
    print("(modelo já está treinado)")
    
    try:
        start_time = time.time()
        
        response = requests.get(
            f"http://localhost:8000/predict/{ticker}",
            params={"days": days},
            timeout=60
        )
        
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✓ Previsão concluída em {elapsed_time:.1f} segundos")
            print(f"   (muito mais rápido que a primeira vez!)")
            print(f"\nPrevisões para {days} dias:")
            for i, pred in enumerate(result['predictions'], 1):
                print(f"  Dia {i}: R$ {pred:.2f}")
            
            return True
        else:
            print(f"\n✗ Erro: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"\n✗ Erro: {e}")
        return False


def test_multiple_tickers():
    """
    Testa previsão para múltiplos tickers
    """
    print("\n\n" + "=" * 70)
    print("TESTE COM MÚLTIPLOS TICKERS")
    print("=" * 70)
    
    tickers = ["VALE3.SA", "ITUB4.SA", "MGLU3.SA"]
    
    print(f"\nTestando {len(tickers)} tickers diferentes")
    print("Cada um treinará automaticamente se necessário\n")
    
    results = {}
    
    for ticker in tickers:
        print(f"\n→ Processando {ticker}...")
        
        try:
            start_time = time.time()
            
            response = requests.get(
                f"http://localhost:8000/predict/{ticker}",
                params={"days": 3},
                timeout=600
            )
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                results[ticker] = {
                    "success": True,
                    "time": elapsed_time,
                    "predictions": result['predictions']
                }
                print(f"  ✓ Concluído em {elapsed_time:.1f}s")
                print(f"  → Previsões: {[f'R$ {p:.2f}' for p in result['predictions']]}")
            else:
                results[ticker] = {
                    "success": False,
                    "error": response.text
                }
                print(f"  ✗ Erro: {response.status_code}")
                
        except Exception as e:
            results[ticker] = {
                "success": False,
                "error": str(e)
            }
            print(f"  ✗ Erro: {e}")
    
    print("\n" + "=" * 70)
    print("RESUMO")
    print("=" * 70)
    
    successful = sum(1 for r in results.values() if r.get("success"))
    print(f"\n✓ {successful}/{len(tickers)} tickers processados com sucesso")
    
    for ticker, result in results.items():
        if result.get("success"):
            print(f"\n{ticker}:")
            print(f"  - Tempo: {result['time']:.1f}s")
            print(f"  - Previsões: {result['predictions']}")
        else:
            print(f"\n{ticker}: ✗ FALHOU - {result.get('error', 'Erro desconhecido')}")
    
    print("=" * 70)
    
    return successful == len(tickers)


def main():
    """
    Executa todos os testes
    """
    print("\n" + "=" * 70)
    print("DEMONSTRAÇÃO DE TREINAMENTO AUTOMÁTICO")
    print("=" * 70)
    print("\nEste script demonstra a funcionalidade de treinamento automático.")
    print("Quando você solicita uma previsão para um ticker sem modelo,")
    print("o sistema automaticamente:")
    print("  1. Detecta que o modelo não existe")
    print("  2. Busca dados do Yahoo Finance")
    print("  3. Treina o modelo")
    print("  4. Faz a previsão")
    print("\nTudo isso de forma transparente!")
    print("\nCertifique-se de que a API está rodando:")
    print("  python api.py")
    print("=" * 70)
    
    input("\nPressione ENTER para iniciar os testes...")
    
    success1 = test_auto_training()
    
    if success1:
        input("\nPressione ENTER para testar previsão em cache...")
        success2 = test_cached_prediction()
    else:
        success2 = False
    
    print("\n")
    if input("Deseja testar múltiplos tickers? (s/n): ").lower() == 's':
        success3 = test_multiple_tickers()
    else:
        success3 = None
    
    print("\n\n" + "=" * 70)
    print("RESUMO FINAL")
    print("=" * 70)
    print(f"Teste 1 (Treinamento automático): {'✓ PASSOU' if success1 else '✗ FALHOU'}")
    print(f"Teste 2 (Previsão em cache): {'✓ PASSOU' if success2 else '✗ FALHOU' if success1 else '- PULADO'}")
    print(f"Teste 3 (Múltiplos tickers): {'✓ PASSOU' if success3 else '✗ FALHOU' if success3 is False else '- PULADO'}")
    print("=" * 70)
    
    print("\n📝 LIÇÕES APRENDIDAS:")
    print("  - Não é mais necessário treinar manualmente antes de prever")
    print("  - O sistema gerencia o treinamento automaticamente")
    print("  - Previsões subsequentes são muito mais rápidas (cache)")
    print("  - Logs detalhados ajudam no debugging")
    print("=" * 70)


if __name__ == "__main__":
    main()
