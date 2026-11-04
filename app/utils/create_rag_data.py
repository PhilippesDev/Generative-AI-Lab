"""Générateur de données pour le RAG (Partie C et D).
Crée :
1. Le corpus documentaire (4 documents de référence, ~4000 mots).
2. Le fichier de 20 questions d'évaluation (15 présentes, 5 absentes) avec la vérité terrain.
"""

import json
from pathlib import Path

DOCS = {
    "reglement_tp_isig.txt": """# RÈGLEMENT OFFICIEL DES TRAVAUX PRATIQUES — ISIG GOMA
Département de Génie Logiciel — Année académique 2026-2027

ARTICLE 1 : DISPOSITIONS GÉNÉRALES
Les travaux pratiques du cours d'Intelligence Artificielle sont obligatoires pour tous les étudiants inscrits en Licence 3 Génie Logiciel. Ils visent à doter l'étudiant de compétences réelles en implémentation d'algorithmes et d'architectures d'IA générative. Tout étudiant inscrit doit se conformer strictement aux directives académiques de l'enseignant Blaise CIZA.

ARTICLE 2 : CONSTITUTION DES GROUPES ET VARIANTES
Les travaux pratiques d'IA se font en groupes de cinq étudiants au maximum. Aucun travail individuel n'est autorisé sans dérogation formelle du bureau décanal. Un groupe peut exceptionnellement mélanger des étudiants de filières différentes, à condition de choisir une variante unique et de la déclarer expressément lors du premier rapport. Pour la filière Génie Logiciel, la variante imposée exige la mise en œuvre d'une architecture orientée services avec FastAPI, d'une interface web réactive, d'un banc de 20 tests automatisés avec pytest et de la tenue d'un journal des requêtes.

ARTICLE 3 : ASSIDUITÉ ET PONCTUALITÉ
La présence physique aux séances de laboratoire à l'ISIG-GOMA est rigoureusement contrôlée au moyen d'une liste d'émargement. Trois retards non justifiés de plus de quinze minutes équivalent à une absence non justifiée. Deux absences non justifiées entraînent automatiquement l'exclusion du groupe et l'attribution de la note zéro pour l'ensemble des TP.

ARTICLE 4 : DÉPÔT DES LIVRABLES ET MODALITÉS D'ÉVALUATION
Chaque groupe doit remettre à l'échéance fixée : une archive zip contenant le code source intégralement commenté, le fichier requirements.txt, un rapport au format PDF comptant entre 5 et 10 pages agrémenté de captures d'écran, ainsi que l'annexe B relative à la déclaration d'usage de l'intelligence artificielle. Les cinq TP semestriels comptent pour 50 pour cent de la note finale du module, les 50 pour cent restants provenant de la soutenance orale du projet d'agent autonome.

ARTICLE 5 : POLITIQUE ANTI-PLAGIAT ET USAGE DE L'IA
Le plagiat entre groupes d'étudiants ou la reprise intégrale d'un code source trouvé en ligne sans citation explicite de la source est sanctionné par la note zéro immédiate. L'usage d'outils d'intelligence artificielle est toléré à des fins pédagogiques d'explication, d'aide au débogage ou de relecture stylistique. Toutefois, chaque groupe est tenu de déclarer en annexe B les outils mobilisés, les requêtes soumises et les vérifications indépendantes conduites par les membres.

ARTICLE 6 : MATÉRIEL ET ENVIRONNEMENT EN LABORATOIRE
Les ordinateurs du laboratoire informatique de l'ISIG fonctionnent sous système d'exploitation Windows 11 avec des environnements virtuels Python 3.10 ou supérieur. L'installation de bibliothèques tierces volumineuses doit obligatoirement être soumise à l'approbation préalable de l'administrateur du réseau académique afin de préserver l'espace disque et la bande passante locale.
""",

    "procedure_deploiement_devops.txt": """# PROCÉDURE DE DÉPLOIEMENT CONTINU ET BONNES PRATIQUES DEVOPS
Direction des Systèmes d'Information — Document Technique Interne

1. INTRODUCTION ET OBJECTIFS OPÉRATIONNELS
Le présent guide standardise l'ensemble des procédures d'intégration et de déploiement continus applicables à nos applications hébergées. Il garantit une cadence de livraison rapide tout en maintenant une stabilité optimale des services en production.

2. GESTION DES BRANCHES SOUS GIT
Nous appliquons strictement la méthodologie Gitflow allégée. La branche 'main' reflète en permanence l'état déployé en environnement de production. La branche 'develop' centralise l'intégration quotidienne des fonctionnalités validées. Toute nouvelle tâche donne lieu à la création d'une branche préfixée 'feature/' issue de 'develop'. Avant toute fusion vers 'develop', une pull request doit obligatoirement obtenir l'approbation formelle de deux relecteurs pairs et satisfaire la totalité des tests unitaires du pipeline.

3. STRATÉGIE DE CONTENEURISATION DOCKER
Toutes les applications logicielles doivent être empaquetées sous forme d'images de conteneurs Docker légères basées sur Alpine Linux ou Debian Slim. Les Dockerfiles doivent appliquer le principe de construction multi-étapes pour minimiser la taille de l'artefact final et restreindre la surface d'attaque. Il est formellement interdit d'exécuter un conteneur avec les privilèges de l'utilisateur root en environnement de production.

4. AUTOMATISATION DES TESTS ET QUALITÉ DE CODE
Le pipeline d'intégration continue exécute automatiquement les étapes suivantes à chaque soumission de code : linter flake8 pour la conformité syntaxique, analyseur statique ruff, exécution des tests unitaires et fonctionnels avec pytest, et calcul de la couverture de test. Un seuil minimal de 80 pour cent de couverture des instructions est exigé pour autoriser le déploiement.

5. STRATÉGIE DE DÉPLOIEMENT ET RETOUR ARRIÈRE
Les déploiements sur les clusters Kubernetes de production utilisent exclusivement la stratégie de mise à jour progressive ou blue-green. Cette approche garantit une disponibilité de service sans aucune interruption pour l'utilisateur final. En cas de dysfonctionnement critique constaté par les sondes de santé, un mécanisme de rollback automatisé restaure la version précédente sous un délai maximal de quatre-vingt-dix secondes.

6. OBSERVABILITÉ, TRACES ET GESTION DES ALERTES
La supervision de nos services repose sur la collecte centralisée des métriques avec Prometheus et la visualisation des tableaux de bord sur Grafana. Les journaux d'événements doivent obligatoirement être sérialisés au format JSON et comporter un identifiant unique de corrélation par requête utilisateur. Les alertes de criticité haute sont immédiatement relayées sur le canal d'astreinte technique pour une prise en charge sous quinze minutes.
""",

    "spec_api_gestion_etudiants.txt": """# SPÉCIFICATION TECHNIQUE DE L'API REST GESTION ÉTUDIANTS
Version 2.4 — Architecture Microservices ISIG

1. DESCRIPTION GÉNÉRALE DU SERVICE
L'API REST Gestion Étudiants centralise les opérations d'inscription, de suivi pédagogique, de délibération et de délivrance des relevés de notes. Elle est exposée à l'adresse sécurisée https://api.isig.ac.cd/v2 et respecte scrupuleusement les principes architecturaux RESTful avec typage Pydantic.

2. AUTHENTIFICATION ET SÉCURITÉ DES ACCÈS
Tous les points d'entrée de l'API requièrent une authentification par jeton JWT transmis au sein de l'en-tête HTTP 'Authorization' avec le préfixe 'Bearer'. La durée de validité d'un jeton d'accès est fixée à soixante minutes. Au-delà, le client doit présenter son jeton de rafraîchissement dont la validité est de quatorze jours. L'accès aux données des étudiants est soumis à un contrôle basé sur les rôles : administrateur, enseignant et étudiant.

3. GESTION DES QUOTAS ET LIMITATION DE DÉBIT
Afin de préserver l'infrastructure contre les attaques par déni de service et les requêtes abusives, un limiteur de débit applique un plafond strict de cent requêtes par minute et par adresse IP pour les utilisateurs non authentifiés, et de mille requêtes par minute pour les partenaires académiques disposant d'une clé d'API dédiée. Tout dépassement retourne le code de statut HTTP 429 Too Many Requests.

4. DESCRIPTION DÉTAILLÉE DES ROUTES PRINCIPALES
- GET /etudiants : retourne la liste paginée des étudiants inscrits. Accepte les paramètres de requête 'page' (défaut 1), 'limit' (maximum 50) et 'filiere'.
- POST /etudiants : enregistre un nouvel étudiant dans la base. Le corps de la requête doit respecter le schéma JSON comportant matricule, nom, prénom, filière et niveau d'études. En cas de succès, l'API renvoie le code HTTP 201 Created avec l'en-tête Location.
- GET /etudiants/{matricule}/releve : génère le relevé de notes au format JSON pour l'année académique courante. Code 200 OK en cas de succès, 404 Not Found si le matricule est inexistant.
- DELETE /etudiants/{matricule} : désactive le compte étudiant sans suppression physique de la base pour des raisons d'archivage réglementaire.

5. FORMAT DES RÉPONSES ET GESTION DES ANOMALIES
Toutes les réponses de l'API sont encodées en UTF-8 et structurées selon le standard JSON. En situation d'erreur, le serveur retourne systématiquement un objet JSON normalisé contenant trois propriétés : un code d'erreur textuel, un message descriptif destiné au développeur et un horodatage ISO 8601.
""",

    "guide_bonnes_pratiques_architecture.txt": """# GUIDE DE BONNES PRATIQUES D'ARCHITECTURE ET TESTS LOGICIELS
Club Génie Logiciel — Fascicule de Référence Technique

1. LES PRINCIPES FONDAMENTAUX DE CONCEPTION SOLID
L'écriture d'un code robuste et évolutif repose sur le respect des cinq piliers SOLID :
- Responsabilité Unique : un module ou une classe ne doit posséder qu'une seule raison de changer.
- Ouvert et Fermé : une entité logicielle doit être ouverte aux extensions mais fermée aux modifications directes.
- Substitution de Liskov : une sous-classe doit pouvoir être substituée à sa classe mère sans altérer l'exactitude du programme.
- Ségrégation des Interfaces : il convient de concevoir plusieurs interfaces restreintes et spécialisées plutôt qu'une interface unique surchargée.
- Inversion des Dépendances : les modules de haut niveau ne doivent pas dépendre des modules de bas niveau, mais tous deux doivent dépendre d'abstractions.

2. PATRONS DE CONCEPTION RECOMMANDÉS
Pour résoudre les problématiques récurrentes de structure et de comportement, nous préconisons l'usage des patrons éprouvés :
- Le patron Fabrique sépare la création des objets de leur utilisation métier, facilitant l'injection de dépendances.
- Le patron Observateur permet d'établir une communication découplée de type un-à-plusieurs lors de la survenue d'événements asynchrones.
- Le patron Stratégie encapsule des familles d'algorithmes interchangeables au moment de l'exécution, très utile pour basculer entre la recherche TF-IDF et la recherche par embeddings.

3. PYRAMIDE DES TESTS LOGICIELS
Une stratégie de qualification efficace équilibre les efforts de vérification selon trois niveaux :
- La base de la pyramide est constituée de tests unitaires rapides et isolés, couvrant au minimum soixante-dix pour cent de la logique métier.
- Le niveau intermédiaire regroupe les tests d'intégration qui vérifient la conformité des interactions entre la base de données, les API tierces et le serveur d'application.
- Le sommet de la pyramide comprend les tests de bout en bout qui automatisent des scénarios d'usage réels à travers l'interface utilisateur.

4. GESTION DES DONNÉES ET CACHES DE HAUTE PERFORMANCE
L'utilisation judicieuse d'une couche de mémoire cache comme Redis permet de soulager le serveur de base de données relationnelle PostgreSQL. Les données statiques ou fréquemment consultées font l'objet d'une mise en cache avec un temps d'expiration adapté afin de préserver la fraîcheur des informations sans pénaliser la latence globale.
"""
}

# 20 Questions : 15 présentes dans les documents, 5 absentes
QUESTIONS_20 = [
    # 15 questions PRÉSENTES
    {
        "id": 1,
        "question": "Combien d'étudiants au maximum peuvent composer un groupe de TP en IA ?",
        "reponse_attendue": "Un groupe de TP est composé de 5 étudiants au maximum.",
        "est_presente": True,
        "source_attendue": "reglement_tp_isig.txt",
        "mots_cles": ["groupe", "étudiants", "maximum", "5", "cinq"]
    },
    {
        "id": 2,
        "question": "Quelle est la pénalité pour deux absences non justifiées aux séances de TP ?",
        "reponse_attendue": "Deux absences non justifiées entraînent l'exclusion du groupe et l'attribution de la note zéro pour l'ensemble des TP.",
        "est_presente": True,
        "source_attendue": "reglement_tp_isig.txt",
        "mots_cles": ["absence", "exclusion", "zéro", "note"]
    },
    {
        "id": 3,
        "question": "Quel pourcentage de la note finale du module représentent les cinq travaux pratiques ?",
        "reponse_attendue": "Les cinq TP comptent pour 50 pour cent de la note finale du module.",
        "est_presente": True,
        "source_attendue": "reglement_tp_isig.txt",
        "mots_cles": ["50", "pour cent", "note finale"]
    },
    {
        "id": 4,
        "question": "Quel est le nombre de pages exigé pour le rapport PDF à remettre ?",
        "reponse_attendue": "Le rapport PDF doit compter entre 5 et 10 pages.",
        "est_presente": True,
        "source_attendue": "reglement_tp_isig.txt",
        "mots_cles": ["5 et 10 pages", "rapport", "PDF"]
    },
    {
        "id": 5,
        "question": "Quelles sont les exigences imposées pour la variante de la filière Génie Logiciel ?",
        "reponse_attendue": "FastAPI, une interface web, 20 tests automatiques pytest et un journal des requêtes.",
        "est_presente": True,
        "source_attendue": "reglement_tp_isig.txt",
        "mots_cles": ["FastAPI", "interface web", "tests", "journal"]
    },
    {
        "id": 6,
        "question": "À quoi correspond la branche 'main' dans la procédure Gitflow appliquée ?",
        "reponse_attendue": "La branche 'main' reflète en permanence l'état déployé en environnement de production.",
        "est_presente": True,
        "source_attendue": "procedure_deploiement_devops.txt",
        "mots_cles": ["main", "production", "déployé"]
    },
    {
        "id": 7,
        "question": "Quel seuil minimal de couverture de code est exigé pour autoriser le déploiement ?",
        "reponse_attendue": "Un seuil minimal de 80 pour cent de couverture des instructions est exigé.",
        "est_presente": True,
        "source_attendue": "procedure_deploiement_devops.txt",
        "mots_cles": ["80", "pour cent", "couverture"]
    },
    {
        "id": 8,
        "question": "En combien de temps le mécanisme de rollback automatisé restaure-t-il la version précédente ?",
        "reponse_attendue": "Le rollback automatisé restaure la version précédente sous un délai maximal de 90 secondes.",
        "est_presente": True,
        "source_attendue": "procedure_deploiement_devops.txt",
        "mots_cles": ["90 secondes", "quatre-vingt-dix", "rollback"]
    },
    {
        "id": 9,
        "question": "Quels outils sont utilisés pour collecter et visualiser les métriques de supervision ?",
        "reponse_attendue": "Prometheus pour la collecte des métriques et Grafana pour la visualisation des tableaux de bord.",
        "est_presente": True,
        "source_attendue": "procedure_deploiement_devops.txt",
        "mots_cles": ["Prometheus", "Grafana"]
    },
    {
        "id": 10,
        "question": "Quelle est la durée de validité d'un jeton d'accès JWT pour l'API REST ?",
        "reponse_attendue": "La durée de validité d'un jeton d'accès est fixée à 60 minutes.",
        "est_presente": True,
        "source_attendue": "spec_api_gestion_etudiants.txt",
        "mots_cles": ["60 minutes", "soixante", "jeton"]
    },
    {
        "id": 11,
        "question": "Quel est le plafond de requêtes autorisé par minute pour un utilisateur non authentifié ?",
        "reponse_attendue": "Le plafond est de 100 requêtes par minute et par adresse IP.",
        "est_presente": True,
        "source_attendue": "spec_api_gestion_etudiants.txt",
        "mots_cles": ["100 requêtes", "cent", "minute"]
    },
    {
        "id": 12,
        "question": "Quel code de statut HTTP est retourné lorsqu'un nouvel étudiant est créé avec succès via POST ?",
        "reponse_attendue": "L'API renvoie le code HTTP 201 Created.",
        "est_presente": True,
        "source_attendue": "spec_api_gestion_etudiants.txt",
        "mots_cles": ["201", "Created"]
    },
    {
        "id": 13,
        "question": "Que stipule le principe de responsabilité unique dans les principes SOLID ?",
        "reponse_attendue": "Un module ou une classe ne doit posséder qu'une seule raison de changer.",
        "est_presente": True,
        "source_attendue": "guide_bonnes_pratiques_architecture.txt",
        "mots_cles": ["responsabilité unique", "seule raison de changer"]
    },
    {
        "id": 14,
        "question": "Quel patron de conception permet d'alterner entre la recherche TF-IDF et les embeddings au moment de l'exécution ?",
        "reponse_attendue": "Le patron Stratégie (Strategy Pattern).",
        "est_presente": True,
        "source_attendue": "guide_bonnes_pratiques_architecture.txt",
        "mots_cles": ["Stratégie", "Strategy"]
    },
    {
        "id": 15,
        "question": "Quel pourcentage minimal de la logique métier doit être couvert par les tests unitaires selon la pyramide des tests ?",
        "reponse_attendue": "Les tests unitaires doivent couvrir au minimum 70 pour cent de la logique métier.",
        "est_presente": True,
        "source_attendue": "guide_bonnes_pratiques_architecture.txt",
        "mots_cles": ["70", "pour cent", "tests unitaires"]
    },

    # 5 questions ABSENTES (Le système doit répondre rigoureusement « Je ne sais pas »)
    {
        "id": 16,
        "question": "Quel est le montant exact des frais d'inscription académique à l'ISIG-GOMA pour l'année 2026-2027 ?",
        "reponse_attendue": "Je ne sais pas.",
        "est_presente": False,
        "source_attendue": None,
        "mots_cles": ["frais d'inscription", "montant"]
    },
    {
        "id": 17,
        "question": "Quelle est l'adresse physique du campus principal de l'ISIG dans la ville de Goma ?",
        "reponse_attendue": "Je ne sais pas.",
        "est_presente": False,
        "source_attendue": None,
        "mots_cles": ["adresse physique", "quartier"]
    },
    {
        "id": 18,
        "question": "Quel est le nom du recteur ou du directeur général actuel de l'ISIG-GOMA ?",
        "reponse_attendue": "Je ne sais pas.",
        "est_presente": False,
        "source_attendue": None,
        "mots_cles": ["directeur général", "recteur"]
    },
    {
        "id": 19,
        "question": "À quelle heure ferme la bibliothèque universitaire le samedi après-midi ?",
        "reponse_attendue": "Je ne sais pas.",
        "est_presente": False,
        "source_attendue": None,
        "mots_cles": ["bibliothèque", "samedi"]
    },
    {
        "id": 20,
        "question": "Quel est le débit garanti de la connexion fibre optique souscrite auprès de l'opérateur Airtel ?",
        "reponse_attendue": "Je ne sais pas.",
        "est_presente": False,
        "source_attendue": None,
        "mots_cles": ["fibre", "débit", "Airtel"]
    }
]


def generer_fichiers(base_dir: Path):
    doc_dir = base_dir / "data" / "documents"
    q_dir = base_dir / "data" / "questions"
    doc_dir.mkdir(parents=True, exist_ok=True)
    q_dir.mkdir(parents=True, exist_ok=True)

    # 1. Écrire les documents
    for filename, content in DOCS.items():
        with open(doc_dir / filename, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f"Document créé : {filename}")

    # 2. Écrire les questions d'évaluation
    with open(q_dir / "benchmark_20_questions.json", "w", encoding="utf-8") as f:
        json.dump(QUESTIONS_20, f, ensure_ascii=False, indent=2)
    print(f"Benchmark de 20 questions généré dans {q_dir / 'benchmark_20_questions.json'}")


if __name__ == "__main__":
    racine = Path(__file__).resolve().parent.parent.parent
    generer_fichiers(racine)
