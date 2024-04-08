NAME=j-desktop

if [ "$1" -eq 1 ]; then
    kubectl delete deployment $NAME
    kubectl delete service $NAME
    kubectl delete ingress $NAME
elif [ "$1" -eq 0 ]; then
    kubectl create -f desktop.yml
    kubectl create -f desktop-ingress.yml
else
    echo "Usage: $0 [1]"
    exit 1
fi