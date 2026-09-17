# Update FCQ data to the latest available and push to S3
update:
    python3 fcq_processor.py --download_and_update

# Deploy awslambda.py to the fcqprocessor Lambda and confirm the deployed
# code hash matches what was just pushed.
deploy_lambda:
    #!/usr/bin/env bash
    set -euo pipefail

    FUNCTION_NAME=fcqprocessor
    REGION=us-east-1
    BUILD_DIR=$(mktemp -d)
    ZIP_PATH="$BUILD_DIR/lambda.zip"

    trap 'rm -rf "$BUILD_DIR"' EXIT

    cp awslambda.py "$BUILD_DIR/lambda_function.py"
    (cd "$BUILD_DIR" && zip -q lambda.zip lambda_function.py)

    LOCAL_SHA=$(openssl dgst -sha256 -binary "$ZIP_PATH" | openssl base64)
    echo "Local code hash:    $LOCAL_SHA"

    echo "Uploading to $FUNCTION_NAME ($REGION)..."
    aws lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --region "$REGION" \
        --zip-file "fileb://$ZIP_PATH" \
        > /dev/null

    echo "Waiting for update to finish applying..."
    while true; do
        STATUS=$(aws lambda get-function-configuration \
            --function-name "$FUNCTION_NAME" \
            --region "$REGION" \
            --query "LastUpdateStatus" --output text)
        if [ "$STATUS" = "Successful" ]; then
            break
        elif [ "$STATUS" = "Failed" ]; then
            echo "Lambda update failed" >&2
            exit 1
        fi
        sleep 2
    done

    DEPLOYED_SHA=$(aws lambda get-function-configuration \
        --function-name "$FUNCTION_NAME" \
        --region "$REGION" \
        --query "CodeSha256" --output text)
    echo "Deployed code hash: $DEPLOYED_SHA"

    if [ "$LOCAL_SHA" != "$DEPLOYED_SHA" ]; then
        echo "MISMATCH: deployed code does not match local awslambda.py" >&2
        exit 1
    fi

    echo "Confirmed: fcqprocessor is now running the local awslambda.py"
