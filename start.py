import subprocess
import time
import requests

def start_server():
    """Lance le serveur FastAPI via fastapi/main.py."""
    print("Lancement du serveur FastAPI...")
    process = subprocess.Popen(["python", "fastapi/main.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)  # Attendre un peu pour que le serveur démarre
    return process

def test_server():
    """Vérifie si le serveur est prêt."""
    print("Vérification que le serveur est actif...")
    url = "http://127.0.0.1:8000"  # Adresse par défaut de FastAPI
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Le serveur est actif.")
            return True
    except requests.exceptions.ConnectionError:
        print("Le serveur n'est pas encore prêt...")
    return False

def start_client():
    """Lance le script client (client.py)."""
    print("Lancement du client...")
    subprocess.run(["python", "fastAPI/client.py"])  # Bloque l'exécution jusqu'à la fin du client

def main():
    server_process = start_server()
    try:
        # Vérification du démarrage du serveur
        for _ in range(5):  # Réessayer plusieurs fois
            if test_server():
                break
            time.sleep(2)
        else:
            print("Le serveur n'a pas démarré correctement.")
            return

        # Lancer le client une fois le serveur prêt
        start_client()
    finally:
        # Arrêter le serveur une fois la démonstration terminée
        print("Arrêt du serveur...")
        server_process.terminate()

if __name__ == "__main__":
    main()
