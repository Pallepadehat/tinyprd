# Ordbog

Korte forklaringer på de fagord, der bruges i projektet. I alfabetisk rækkefølge.

| Ord | Forklaring |
|---|---|
| **Attention** | Den mekanisme i en transformer, hvor hver position "kigger på" andre positioner og henter information fra dem. |
| **Attention-maske** | En tabel med sandt/falsk, der bestemmer, hvem der må kigge på hvem. |
| **Autoregressiv (AR)** | En model, der skriver ét ord ad gangen, hvor hvert nyt ord afhænger af alle de forrige. Sådan virker fx ChatGPT. |
| **Batch** | En lille bunke eksempler, som modellen træner på samtidig. |
| **Blok-diagonal maske** | En attention-maske, hvor hvert niveau kun må se idéen og sig selv. Se [SAADAN_VIRKER_DET.md](SAADAN_VIRKER_DET.md). |
| **Checkpoint** | En gemt fil med en trænet models vægte. Ligger i `runs/`. |
| **Codebook / kodebog** | En tabel med fx 512 vektorer. Hver vektor har et nummer, en *kode*. |
| **Corruption / forstyrrelse** | Under træning byttes nogle koder ud med tilfældige, så modellen lærer at tåle sine egne fejl. |
| **Cross-entropy** | Det mest almindelige loss til klassifikation. Det er lavt, når modellen giver det rigtige svar høj sandsynlighed. |
| **Decoder** | Den del, der laver en intern repræsentation om til output, her ord. |
| **Diffusionsmodel** | En model, der starter med støj og forfiner *hele* teksten mange gange. NCM er et alternativ. |
| **Embedding** | En vektor, der repræsenterer et ord (eller en position) som tal. |
| **Encoder** | Den del, der laver input, her ord, om til en intern repræsentation. |
| **Epoch** | Én gennemgang af hele træningsdatasættet. |
| **Forward pass** | Ét kald af modellen: input ind, output ud. |
| **Generalisering** | At modellen også klarer sig godt på eksempler, den aldrig har set. |
| **Gradient** | Retningen, vægtene skal skubbes i, for at loss bliver lavere. |
| **Greedy** | At man altid vælger det mest sandsynlige svar (temperatur 0). |
| **Latent vektor** | En intern vektor, som modellen selv har lært. Den er ikke direkte et ord. |
| **Logits** | Modellens rå scorer før de laves om til sandsynligheder. |
| **Loss** | Et tal for, hvor forkert modellen er. Træning prøver at gøre det lavt. |
| **MPS** | *Metal Performance Shaders*, Apples GPU-backend til PyTorch på M1/M2/M3-Macs. |
| **NCM** | *Next-Concept Modeling*. Generering fra groft til fint, niveau for niveau. |
| **Nested dropout** | Under træning ser decoderen nogle gange kun de første niveauer, så de grove niveauer tvinges til at være meningsfulde. |
| **NFE** | *Number of Function Evaluations*: antal forward passes for at lave ét output. Et mål for hastighed. |
| **Niveau** | Ét lag i hierarkiet. Niveau 1 har 2 koder (groft), niveau 8 har 256 (ét pr. ord). |
| **Overfitting** | Når modellen lærer træningsdata udenad i stedet for det generelle mønster. |
| **Parameter** | Et enkelt tal (en vægt) i modellen, som justeres under træning. |
| **PRD** | *Product Requirements Document*. Et kort dokument, der forklarer problem, mål og afgrænsning. Se [PRD_PRINCIPPER.md](PRD_PRINCIPPER.md). |
| **Prompt** | Det input, modellen får, her den rodede idé. |
| **Residual** | "Resten": det, der stadig mangler at blive beskrevet. |
| **Residual-kvantisering** | At beskrive en vektor med flere koder efter hinanden, hvor hver kode beskriver det, de forrige manglede. |
| **Straight-through** | Et trick, der lader gradienten passere en ikke-differentiabel operation (som "vælg nærmeste kode"), som om den ikke var der. |
| **Syntetisk data** | Data, der er genereret af et program i stedet for at være indsamlet. |
| **Temperatur** | Styrer, hvor tilfældig sampling er. 0 er deterministisk, højere giver mere variation. |
| **Token** | Én enhed tekst. I TinyPRD er det ét ord, ét tegn eller ét linjeskift. |
| **Transformer** | Den neurale netværksarkitektur, der ligger bag næsten alle moderne sprogmodeller. |
| **VQ-VAE** | *Vector-Quantized Variational AutoEncoder*. En autoencoder, der beskriver input med koder fra en kodebog. |
