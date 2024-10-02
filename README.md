# mishima-fastapi-samples

## Description

Mishima.syk #21 で発表したFastAPIのサンプルコードです。

## Usage

`reinvent-backend`の実行には公式リポジトリのソースコードが必要です。[MolecularAI/REINVENT4](https://github.com/MolecularAI/REINVENT4/tree/main)をクローンして`reinvent-backend`ディレクトリに配置してください。

- localでの実行

Dockerを使ってローカルで実行する場合はそれぞれのディレクトリに移動したうえで、次のコマンドを実行してください。

```bash
docker build -t IMAGE_NAME .
docker run -d --name CONTAINER_NAME -p 8080:8080 IMAGE_NAME
```

- Google CloudのCloud Build でイメージビルドして Cloud Run へデプロイする場合

Google Cloudでプロジェクトを作成し、利用するAPIを有効化したうえで、gCloud CLIを利用して次のコマンドを実行。

```bash
# コンテナをビルドして、Container Registry にプッシュ
gcloud builds submit --region=asia-east1 --tag asia-northeast1-docker.pkg.dev/PROJECT_NAME/containers/IMAGE_NAME:TAG

# Cloud Run にデプロイ。バックエンドはメモリを8Gi、cpuを4くらいに設定するといい
gcloud run deploy DEPLOYMENT_NAME \
    --image=asia-northeast1-docker.pkg.dev/PROJECT_NAME/REGISTRY_NAME/IMAGE_NAME:TAG \
    --memory=1Gi \
    --cpu=1 \
    --service-account=ROLE_NAME@PROJECT_NAME.iam.gserviceaccount.com \
    --region=asia-northeast1

# reinvent frontend に対しては、backendのURLを環境変数REINVENT_API_URLとして渡す必要があります。
gcloud run deploy DEPLOYMENT_NAME \
    --image=asia-northeast1-docker.pkg.dev/PROJECT_NAME/REGISTRY_NAME/IMAGE_NAME:TAG \
    --memory=1Gi \
    --cpu=1 \
    --service-account=ROLE_NAME@PROJECT_NAME.iam.gserviceaccount.com \
    --region=asia-northeast1 \
    --update-env-vars REINVENT_API_URL=BACKEND_URL/sampling
```