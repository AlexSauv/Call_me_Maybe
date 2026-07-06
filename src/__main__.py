from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np


def main() -> None:
    print("--- Initialisation du modèle Qwen 0.6B ---")
    # 1. Charger le modèle (gère automatiquement CPU/GPU)
    model = Small_LLM_Model()

    # 2. Définir un prompt de test
    prompt = "What is the sum of 2 and 3? The answer is"
    print(f"Prompt initial : '{prompt}'")

    # 3. Encodage du texte initial via l'SDK
    # .encode() renvoie un tenseur 2D, on l'extrait en liste simple de int
    input_ids_tensor = model.encode(prompt)
    input_ids: list[int] = input_ids_tensor[0].tolist()

    print(f"Tokens IDs initiaux : {input_ids}")

    # 4. Boucle autoregressive : Générer 5 tokens de texte brut
    print("\nGénération en cours...", flush=True)
    for _ in range(5):
        # Récupérer les logits du prochain token (liste de float)
        logits = model.get_logits_from_input_ids(input_ids)

        # Prendre le token avec la probabilité maximale (Greedy search)
        next_token_id = int(np.argmax(logits))

        # Ajouter le token généré à notre historique pour la suite
        input_ids.append(next_token_id)

    # 5. Décodage du résultat final
    texte_final = model.decode(input_ids)
    print("\n--- Résultat de la génération brute ---")
    print(texte_final)


if __name__ == "__main__":
    main()



    
if __name__=="__main__":
    # import os
    # import llm_sdk

    # print("--- DIAGNOSTIC LLM_SDK ---")
    # # 1. Liste les fichiers dans le dossier llm_sdk
    # print("Fichiers trouvés dans llm_sdk :")
    # print(os.listdir(llm_sdk.__path__[0]))

    # print("\nÉléments exportés par llm_sdk :")
    # # 2. Liste ce que Python voit à l'intérieur du package
    # print(dir(llm_sdk))
    # print("--------------------------")
    main()