for f in ~/data/; do
	python to_grayscale.py $f;
done

nohup python VGG16.py &
