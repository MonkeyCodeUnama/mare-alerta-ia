# API de previsão de chuva — Belém

Este serviço recebe dados meteorológicos em JSON e retorna a estimativa do modelo para a ocorrência de precipitação na próxima hora. A FastAPI serve o modelo treinado pelo notebook; ela não treina o modelo nem consulta o provedor meteorológico.

## Arquivos necessários

Depois de executar `IA_Chuva_Belem.ipynb`, copie do arquivo `saida_ia_chuva/fastapi_chuva.zip`:

- `model_bundle.joblib` para `artifacts/model_bundle.joblib`;
- `metadata.json` para `artifacts/metadata.json`;
- `requirements.txt` para a raiz do projeto, substituindo o arquivo genérico;
- `example_request.json` para a raiz, para testar a conexão.

Use o `rain_runtime.py` e o `app.py` que vieram neste projeto. Mantenha a versão de Python usada no treinamento, informada em `metadata.json`. O Dockerfile deste projeto parte de Python 3.12; ajuste a imagem se o treinamento usou outra versão.

Estrutura esperada:

    fastapi_chuva/
    ├── app.py
    ├── rain_runtime.py
    ├── requirements.txt
    ├── example_request.json
    └── artifacts/
        ├── model_bundle.joblib
        └── metadata.json

## Executar localmente

Abra um terminal na pasta do projeto e crie o ambiente virtual:

    python -m venv .venv

Ative-o no PowerShell do Windows:

    .venv\Scripts\Activate.ps1

No Linux ou macOS:

    source .venv/bin/activate

Instale as dependências e inicie o serviço:

    python -m pip install -r requirements.txt
    python -m uvicorn app:app --host 127.0.0.1 --port 8000

A documentação interativa fica em `http://127.0.0.1:8000/docs`. O endpoint `GET /health` confirma que o modelo foi carregado.

## Conectar o backend Java

O backend Java faz uma requisição HTTP `POST` para `http://127.0.0.1:8000/v1/predict` quando Java e FastAPI executam na mesma máquina. Envie `Content-Type: application/json` e o corpo JSON descrito na seção seguinte. O Java deve obter as variáveis meteorológicas no provedor, montar o objeto da requisição e enviar os dados para a FastAPI. O aplicativo cliente continua conversando com o backend Java.

O exemplo abaixo usa o JDK 11 ou superior e Jackson. Adicione Jackson ao projeto Java; em um projeto Maven, a dependência `jackson-databind` é suficiente. Para testar a conexão, ele envia o `example_request.json` exportado pelo notebook:

    import com.fasterxml.jackson.databind.JsonNode;
    import com.fasterxml.jackson.databind.ObjectMapper;
    import java.net.URI;
    import java.net.http.HttpClient;
    import java.net.http.HttpRequest;
    import java.net.http.HttpResponse;
    import java.nio.file.Files;
    import java.nio.file.Path;
    import java.time.Duration;

    ObjectMapper mapper = new ObjectMapper();
    HttpClient client = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(5))
        .build();

    // Exemplo de teste: este arquivo contém uma amostra histórica do conjunto de teste.
    // Em produção, serialize aqui o JSON montado com os dados meteorológicos atuais.
    String requestJson = Files.readString(Path.of("example_request.json"));

    HttpRequest request = HttpRequest.newBuilder()
        .uri(URI.create("http://127.0.0.1:8000/v1/predict"))
        .timeout(Duration.ofSeconds(10))
        .header("Content-Type", "application/json")
        .POST(HttpRequest.BodyPublishers.ofString(requestJson))
        .build();

    HttpResponse<String> response = client.send(
        request,
        HttpResponse.BodyHandlers.ofString()
    );

    if (response.statusCode() != 200) {
        throw new IllegalStateException(
            "FastAPI respondeu HTTP " + response.statusCode() + ": " + response.body()
        );
    }

    JsonNode result = mapper.readTree(response.body());
    double probability = result.path("rain_probability").asDouble();
    System.out.println("Probabilidade estimada: " + (probability * 100) + "%");
    System.out.println("Janela prevista: "
        + result.path("window_start").asText() + " até "
        + result.path("window_end").asText());

Para usar dados atuais, substitua o conteúdo estático de `example_request.json` por um objeto Java serializado com Jackson. O esquema completo está nesse arquivo de exemplo e na lista `required_raw_features` do `metadata.json`. Envie todas as chaves exigidas, as unidades correspondentes, o ponto de Belém e `feature_time` com fuso horário. Valores ausentes conhecidos devem ser enviados como `null`; não omita as chaves. As unidades, o horizonte e o timestamp precisam corresponder ao contrato do modelo.

A resposta contém `rain_probability` entre 0 e 1, `rain_expected`, a janela prevista, horário da inferência e versões do modelo e dos dados. Para exibir porcentagem, multiplique `rain_probability` por 100. `HTTP 422` indica dados ou contrato inválidos; respostas `5xx` indicam uma falha no serviço. Configure timeout e trate esses erros no Java.

### Java e Docker

Se Java executar no computador host e o Docker publicar a porta local, use `http://127.0.0.1:8000`. Se Java também estiver em um contêiner na mesma rede Docker, use `http://weather-model:8000`. `localhost` dentro do contêiner Java aponta para o próprio contêiner, não para a FastAPI.

O `compose.yaml` publicado restringe a porta ao computador local. Não exponha o serviço publicamente sem rede privada e autenticação apropriadas.

## Testar manualmente

Com a API ligada e `example_request.json` na pasta atual:

    curl -X POST http://127.0.0.1:8000/v1/predict -H "Content-Type: application/json" --data-binary @example_request.json

O exemplo exportado usa uma linha histórica de teste. Ele verifica a comunicação com a FastAPI, mas não representa uma previsão atual.

## Escopo e limitações do MVP

O modelo estima precipitação acima do limiar definido no notebook para o intervalo da próxima hora, no ponto de referência de Belém. O alvo de treinamento é precipitação estimada pelo GFS, não observação de uma estação. A validação apresentada ainda não comprova a chance real de chuva, e o modelo não prevê alagamento nem volume de precipitação. Informe esses limites ao exibir a previsão.
