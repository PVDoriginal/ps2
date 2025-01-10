# Proiect Probabilitati & Statistica 2

Scriu cat mai scurt si mai tehnic, Matei pls editeaza (si mai adauga exemple daca vrei)

## Disclaimer
Toate imaginile date ca exemplu sunt din domeniul public, luate de pe net. 

## Ce face? 

Proiectul se bazeaza pe aplicarea unui filtru Gaussian pe o imagine in diferite moduri. 

- Cea mai simpla operatie este sa bluram o imagine: 

Sorina Predut Normala             |  Sorina Predut Blurata
:-------------------------:|:-------------------------:
![Sorina-Nicoleta-Predut](/project/assets/Sorina-Nicoleta-Predut.png) | ![Sorina-Nicoleta-Predut_smoothened](/project/assets/Sorina-Nicoleta-Predut_smoothened.png)

- Datorita proprietatilor filtrului (detalii tehnice mai jos) putem sa si comprimam o imagine in timp ce il aplicam:

 Andrei Sipos Normal - 1021 KB         |  Andrei Sipos Comprimat - 125 KB
:-------------------------:|:-------------------------:
<img src="/project/assets/sipos.jpg" alt="Alt Text" style="width:40em; height:auto;"> | <img src="/project/assets/siposcompressed.png" alt="Alt Text" style="width:40em; height:auto;">


- In exemplul de mai sus, am aplicat filtrul gaussian, iata ce se intampla daca o comprimam fara a aplica filtrul:
  
 Andrei Sipos Comprimat - 125 KB        |  Andrei Sipos Comprimat (Deviatie 0) - 131 KB
:-------------------------:|:-------------------------:
<img src="/project/assets/siposcompressed.png" alt="Alt Text" style="width:40em; height:auto;"> | <img src="/project/assets/siposcompressed_pure.png" alt="Alt Text" style="width:40em; height:auto;">

- Filtrul mai poate fi folosit si la imbunatatirea calitatii imaginilor in momentul in care le marim:
 
 Cezara Benegui Normala - 250 x 250        |  Cezara Benegui Expandata - 2500 x 2500
:-------------------------:|:-------------------------:
<img src="/project/assets/cezara.jpeg" alt="Alt Text" style="width:40em; height:auto;"> | <img src="/project/assets/cezara_expanded.png" alt="Alt Text" style="width:40em; height:auto;">

- Mai jos am folosit un neural network deja antrenat pe facial recognition, pentru a selecta bucati din imagini si a le aplica filtrul:

 Mihai Bucataru Normal        |  Mihai Bucataru Ascuns
:-------------------------:|:-------------------------:
<img src="/project/assets/bucataru.png" alt="Alt Text" style="width:40em; height:auto;"> | <img src="/project/assets/bucataru_masked.png" alt="Alt Text" style="width:40em; height:auto;">

- Cu acelasi model putem selecta si alte parti ale corpului: 
 
 Paul Irofti Normal        |  Paul Irofti Fara Ochi
:-------------------------:|:-------------------------:
<img src="/project/assets/irofti.jpg" alt="Alt Text" style="width:40em; height:auto;"> | <img src="/project/assets/irofti_masked_eyes.png" alt="Alt Text" style="width:40em; height:auto;">

 Moisil Normal        |  Moisil Igrasiat
:-------------------------:|:-------------------------:
<img src="/project/assets/moisil.jpg" alt="Alt Text" style="width:40em; height:auto;"> | <img src="/project/assets/moisil_masked_eyes_mouth.png" alt="Alt Text" style="width:40em; height:auto;">
## Explicatii Cod (ma doare capul)


