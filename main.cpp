#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;



class DataTraining
{
public:
    DataTraining(const string nomFichier);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void resetEof(void) { m_trainingDataFile.clear();}
    void getTopologie(vector<unsigned> &topologie);
    void openFichier(const string nomFichier);

    // Retourne le nombre de données d'entrées lues dans le fichier
    unsigned getProchaineEntree(vector<double> &valeursEntree);
    unsigned getCibleSorties(vector<double> &valeursCibleSortie);
    void closeFichier(void);

private:
    ifstream m_trainingDataFile;
};

void DataTraining::getTopologie(vector<unsigned> &topologie)
{
    string ligne;
    string label;

    getline(m_trainingDataFile, ligne);
    stringstream ss(ligne);
    ss >> label;
    if (this->isEof() || label.compare("topologie:") != 0) {
            cout << "lol";
        abort();
    }

    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topologie.push_back(n);
    }

    return;
}

void DataTraining::closeFichier(void){
    m_trainingDataFile.close();
}

void DataTraining::openFichier(const string nomFichier){
    m_trainingDataFile.open(nomFichier.c_str());
}

DataTraining::DataTraining(const string nomFichier)
{
    m_trainingDataFile.open(nomFichier.c_str());
}

unsigned DataTraining::getProchaineEntree(vector<double> &valeursEntree)
{
    valeursEntree.clear();

    string ligne;
    getline(m_trainingDataFile, ligne);
    stringstream ss(ligne);

    string label;
    ss>> label;
    if (label.compare("in:") == 0) {
        double uneValeur;
        while (ss >> uneValeur) {
            valeursEntree.push_back(uneValeur);
        }
    }

    return valeursEntree.size();
}

unsigned DataTraining::getCibleSorties(vector<double> &valeursCibleSortie)
{
    valeursCibleSortie.clear();

    string ligne;
    getline(m_trainingDataFile, ligne);
    stringstream ss(ligne);

    string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double uneValeur;
        while (ss >> uneValeur) {
            valeursCibleSortie.push_back(uneValeur);
        }
    }

    return valeursCibleSortie.size();
}


struct Connexion
{
    double poids;
    double poidsDelta;
};


class Neurone;

typedef vector<Neurone> Couche;

// ****************** classe Neurone ******************
class Neurone
{
public:
    Neurone(unsigned nbSorties, unsigned monIndex);
    void setValeurSortie(double val) { m_valeurSortie = val; }
    double getValeurSortie(void) const { return m_valeurSortie; }
    void feedForward(const Couche &couchePrecedente);
    void calcGradientsSortie(double valeurCible);
    void calcGradientsCache(const Couche &coucheSuivante);
    void updatePoidsEntree(Couche &couchePrecedente);

private:
    static double eta;   // [0.0..1.0] Taux d'apprentissage global
    static double alpha; // [0.0..n] Multiplicateur du dernier changement de poids (momentum)
    static double fonctionActivation(double x);
    static double fonctionActivationDerivative(double x); // Pour l'apprentissage par rétro-propagation
    static double poidsRandom(void) { return rand() / double(RAND_MAX); } // Retourne un nombre entre 0 et 1
    double sommeDOW(const Couche &coucheSuivante) const;
    double m_valeurSortie;
    vector<Connexion> m_poidsSortie;
    unsigned m_monIndex;
    double m_gradient;
};

double Neurone::eta = 0.3;    // 0.0 lent, 0.2 moyen ..
double Neurone::alpha = 0.3;   // 0.0 pas d'inertie, 0.5 moyenne inertie


void Neurone::updatePoidsEntree(Couche &couchePrecedente)
{
    // Les poids devant être mis à jour sont dans Connexion
    // Au sein des neuronees de la couche précédente

    for (unsigned n = 0; n < couchePrecedente.size(); ++n) {
        Neurone &neurone = couchePrecedente[n];
        double poidsAncienDelta = neurone.m_poidsSortie[m_monIndex].poidsDelta;

        double nouveauPoidsDelta =
                // Entrée individuelle, amplifiée par le gradient et le taux d'entrainement
                eta
                * neurone.getValeurSortie()
                * m_gradient
                // Ajoute un terme d'inertie (momentum), Optimise la descente de la fonction d'erreur
                // Terme d'inertie = une Fraction n du poids mis à jour précédement.
                + alpha
                * poidsAncienDelta;

        neurone.m_poidsSortie[m_monIndex].poidsDelta = nouveauPoidsDelta;
        neurone.m_poidsSortie[m_monIndex].poids += nouveauPoidsDelta;
    }
}

double Neurone::sommeDOW(const Couche &coucheSuivante) const
{
    double somme = 0.0;

    // Somme des noeuds nourris
    for (unsigned n = 0; n < coucheSuivante.size() - 1; ++n) {
        somme += m_poidsSortie[n].poids * coucheSuivante[n].m_gradient;
    }

    return somme;
}

void Neurone::calcGradientsCache(const Couche &coucheSuivante)
{
    double dow = sommeDOW(coucheSuivante);
    m_gradient = dow * Neurone::fonctionActivationDerivative(m_valeurSortie);
}

void Neurone::calcGradientsSortie(double valeurCible)
{
    double delta = valeurCible - m_valeurSortie;
    m_gradient = delta * Neurone::fonctionActivationDerivative(m_valeurSortie);
}

double Neurone::fonctionActivation(double x)
{
    // tanh - rang de sortie [-1.0..1.0]

    return tanh(x/2);
}

double Neurone::fonctionActivationDerivative(double x)
{
    // dérivée de tanh par x = 1 - tanh²x
    return 1.0 - x * x; // Approximation
}

void Neurone::feedForward(const Couche &couchePrecedente)
{
    double somme = 0.0; // Sortie = f(^Si Ii Wi)

    // Fait la somme des sorties des couches précédentes (qui sont les entrées)
    // Inclut le biais de la couche précédente
    for (unsigned n = 0; n < couchePrecedente.size(); ++n) {
        somme += couchePrecedente[n].getValeurSortie() *
                couchePrecedente[n].m_poidsSortie[m_monIndex].poids;
    }

    m_valeurSortie = Neurone::fonctionActivation(somme); // Fonction d'activation
}

Neurone::Neurone(unsigned nbSorties, unsigned monIndex)
{
    for (unsigned c = 0; c < nbSorties; ++c) { // Boucle sur les connexions
        m_poidsSortie.push_back(Connexion()); // Ajoute une connexion
        m_poidsSortie.back().poids = poidsRandom();
    }

    m_monIndex = monIndex;
}


// ****************** classe Net ******************
class Net
{
public:
    Net(const vector<unsigned> &topologie); // Topologie du réseau type: {3,3,3,2}, 3 neuronees d'entrée, 2 couches cachées de 3 neuronees, et 2 neuronees couche de sortie
    void feedForward(const vector<double> &valeursEntree);
    void retroPropagation(const vector<double> &valeurCibles);
    void getResultats(vector<double> &valeursResultat) const;
    double getErreurMoyenneRecente(void) const { return m_erreurMoyenneRecente; }

private:
    vector<Couche> m_couches; // m_couches[numCouche][numNeurone]
    double m_erreur;
    double m_erreurMoyenneRecente;
    static double m_FacteurLissageMoyenRecent;
};


double Net::m_FacteurLissageMoyenRecent = 0.0;


void Net::getResultats(vector<double> &valeursResultat) const
{
    valeursResultat.clear();

    // Boucle sur la couche de sortie
    for (unsigned n = 0; n < m_couches.back().size() - 1; ++n) {
        valeursResultat.push_back(m_couches.back()[n].getValeurSortie());
    }
}

void Net::retroPropagation(const vector<double> &valeurCibles)
{
    // On a besoin:
    // Erreur globale du réseau  (RMS ou Erreur Quadratique Moyenne[Voir wikipedia]
        // On boucle sur les neuronees de sortie (Sauf biais)
        Couche &coucheSortie = m_couches.back();
        m_erreur = 0.0;

        for (unsigned n = 0; n < coucheSortie.size() - 1; ++n) {
            // rms = Racine de Somme de (cible-actuel)²
            double delta = valeurCibles[n] - coucheSortie[n].getValeurSortie(); // Delta: (cible-actuel)
            m_erreur += delta * delta; // Somme (delta)²
        }
        m_erreur /= coucheSortie.size() - 1; // Valeur moyenne de la somme
        m_erreur = sqrt(m_erreur); // Racine de la somme -> RMS

    // Calcule une moyenne récente

    m_erreurMoyenneRecente =
            (m_erreurMoyenneRecente * m_FacteurLissageMoyenRecent + m_erreur)
            / (m_FacteurLissageMoyenRecent + 1.0);

    // Calcule les gradients de la couche de sortie

    for (unsigned n = 0; n < coucheSortie.size() - 1; ++n) {
        coucheSortie[n].calcGradientsSortie(valeurCibles[n]);
    }

    // Calcule les gradients de/des couche(s) cachée(s)

    for (unsigned numCouche = m_couches.size() - 2; numCouche > 0; --numCouche) {
        Couche &coucheCachee = m_couches[numCouche];
        Couche &coucheSuivante = m_couches[numCouche + 1];

        for (unsigned n = 0; n < coucheCachee.size(); ++n) {
            coucheCachee[n].calcGradientsCache(coucheSuivante);
        }
    }

    // Pour toutes les couches; de la couche de sortie à la première couche cachée:
    // Mise à jour des poids:
    for (unsigned numCouche = m_couches.size() - 1; numCouche > 0; --numCouche) {
        Couche &couche = m_couches[numCouche];
        Couche &couchePrecedente = m_couches[numCouche - 1];

        for (unsigned n = 0; n < couche.size() - 1; ++n) {
            couche[n].updatePoidsEntree(couchePrecedente);
        }
    }
}

void Net::feedForward(const vector<double> &valeursEntree)
{
    // Vérifie que le nombre de données entrées est égal au nombre de neuronees de la couche cachée
    assert(valeursEntree.size() == m_couches[0].size() - 1); // -1 pour ne pas compter le biais

    // Assigne les valeurs d'entrée à leur neuronee d'entrée respectif:
    for (unsigned i = 0; i < valeursEntree.size(); ++i) {
        m_couches[0][i].setValeurSortie(valeursEntree[i]); // Met le neuronee i de la couche d'entrée avec la valeur d'entrée correspondante
    }

    // Propagation type Forward, parcours chaque neuronee de chaque couches (A partir de la couche cachée
    // car la couche d'entrée est déjà effectuée)
    for (unsigned numCouche = 1; numCouche < m_couches.size(); ++numCouche) {
        Couche &couchePrecedente = m_couches[numCouche - 1];
        for (unsigned n = 0; n < m_couches[numCouche].size() - 1; ++n) {
            m_couches[numCouche][n].feedForward(couchePrecedente);
        }
    }
}

// Création du réseau
Net::Net(const vector<unsigned> &topologie)
{
    // Récupère le nombre de couches
    unsigned numCouches = topologie.size();
    for (unsigned numCouche = 0; numCouche < numCouches; ++numCouche) {
        m_couches.push_back(Couche()); // Ajoute une couche
        unsigned nbSorties = numCouche == topologie.size() - 1 ? 0 : topologie[numCouche + 1];

        // Une nouvelle couche est créee, maintenant on la remplit avec des neuronees
        // Et on ajoute un biais:
        for (unsigned numNeurone = 0; numNeurone <= topologie[numCouche]; ++numNeurone) { // "<=" pour l'ajout du biais
            m_couches.back().push_back(Neurone(nbSorties, numNeurone)); // Ajoute un neuronee à la dernière couche ajoutée
            cout << "Ajout d'un neuronee!" << endl;
        }

        // Force la valeur de sortie du biais à 1.0
        m_couches.back().back().setValeurSortie(1.0);
    }
}


void afficheVecteurs(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}


int main()
{
    const string nomFichier = "H:/donnees.txt";
    DataTraining trainData(nomFichier);

    // e.g., { 3, 2, 1 }
    vector<unsigned> topologie;

    trainData.getTopologie(topologie);

    Net mNet(topologie);

    vector<double> valeursEntreeA, valeursCiblesA;
    vector<double> valeursEntree,  valeurCibles, valeursResultat;
    int trainingPass = 676;
    int epoques      = 0;
    int epoquesMax   = 2;
    int intermed     = 0;
    int epoquesinter = 0;

    do{

        for(int i = 0; i <= trainingPass; i++){

            //cout << endl << "Enregistrement " << i << endl;
            //afficheVecteurs("ValeursEntree: ", valeursEntree);
            // Récupère les données et lance le feed forward:
            if (trainData.getProchaineEntree(valeursEntree) != topologie[0]) {
                //afficheVecteurs("ValeursEntree: ", valeursEntree);
                break;
            }

            //afficheVecteurs("Entrees:", valeursEntree);
            mNet.feedForward(valeursEntree);

            // Collecte les poids de sortie actuels:
            mNet.getResultats(valeursResultat);
            //afficheVecteurs("Sortie:", valeursResultat);

            // Entraine le réseaux sur ce à quoi devrais ressembler la sortie:
            trainData.getCibleSorties(valeurCibles);
            //afficheVecteurs("Sortie cible:", valeurCibles);
            assert(valeurCibles.size() == topologie.back());

            mNet.retroPropagation(valeurCibles);

            // Retourne comment se déroule l'apprentissage du réseau:
            if(intermed == 1){
                cout << endl << "Enregistrement " << i << endl;
                afficheVecteurs("Entrees:", valeursEntree);
                afficheVecteurs("Sortie:", valeursResultat);
                afficheVecteurs("Sortie cible:", valeurCibles);
                cout << "Erreur moyenne du reseau: "
                 << mNet.getErreurMoyenneRecente() << endl;

            }

        }
        if(intermed == 1) intermed = 0;

        epoques++;
        epoquesinter++;

        if(epoquesinter == 1){
            cout << "Epoque:" << epoques << endl;
            epoquesinter = 0;
        }

        intermed++;




        trainData.closeFichier();
        trainData.resetEof();
        trainData.openFichier(nomFichier);
        trainData.getTopologie(topologie);

    }while(epoques < epoquesMax);
    //Ajout implémentation des époques:
    system("PAUSE");
}

