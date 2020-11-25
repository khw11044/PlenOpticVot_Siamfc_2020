# SiamFC PlenOptic Image 데이터 구조로 트래킹

<pre>
<code>
pip install -r requirements.txt
</pre>
</code>
### Environment
Window os
Anaconda
### Installation
Install Anaconda, then install dependencies:
<pre>
<code>
# install PyTorch >= 1.0
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
# intall OpenCV using menpo channel (otherwise the read data could be inaccurate)
conda install -c menpo opencv
# install GOT-10k toolkit
pip install got10k
conda install scipy
GOT-10k toolkit is a visual tracking toolkit that implements evaluation metrics and tracking pipelines for 9 popular tracking datasets.
</pre>
</code>
### Data
데이터는 프레임별 폴더로 구성되어있고 각 프레임폴더는 숫자로 구분되어 있습니다.(000~773)

각 프레임 폴더는 images폴더와 focal폴더로 구성되어있으며 본 프로젝트는 focal폴더의 이미지를 트래킹하여 images폴더의 이미지를 통해 bounding box를 보여줍니다.
### Running the demo
1.	Setup the sequence path in Final/demo.py 
2.	Setup the checkpoint path of your pretrained model. Default is pretrained/siamfc_alexnet_e50.pth.
3.	Argument 

-model : pretrained 폴더에 학습된 모델이 있습니다.

-vn : tracking할 폴더위치를 지정합니다.

-lf : 각 프레임별images의 이미지 번호를 지정합니다.

-D : 2D tracking의 경우 2D를 3D tracking을 할경우 3D를 입력합니다.

4.	Run:
<pre>
<code>
python Final/demo.py –model ../pretrained –vn ../data/NonVideo4 –lf 005 –D 2D
</pre>
</code>


## 0. 데이터 구조

data -- 000

       -- 001
       
       -- 002 -- focal 
       
       -- ...   ㄴ image

focal 폴더에 101개의 PlenOptic focal 이미지가 있습니다.
image 폴더에는 9개의 일단 이미지가 있습니다.


## + 최종 Final

### 2D와 3D Tracking 

demo.py에서 2D 트래킹과 vot_siamfc_3D_v2_2를 선택적 수행이 가능합니다.

<pre>
<code>
python demo.py --model=../pretrained -vn ../data/NonVideo4_0 -lf 005 -D 2D
</pre>
</code>

명령행 옵션으로 --model, -vn, -lf, -D를 인자값으로 받습니다.

--model은 pretrained폴더에 이미학습된 siamfc 모델

-vn은 읽어들일 plenoptic구조 데이터폴더

-lf는 pletoptic구조의 데이터에서 일반 이미지의 번호 입니다. (예: 모든 이미지의 images폴더의 005.png 파일들만 읽습니다.)  

-D는 2D tracking을 할려면 2D, 선명도 + 3D tracking을 할려면 3D를 입력하면 됩니다.

---------------------------------------------------------------

## 1. vot_siamfc_2D

<div>
<img width="800" src="https://user-images.githubusercontent.com/51473705/99151307-91c27400-26dd-11eb-9a99-fdf9f7217713.PNG">       
</div>

### 2D 이미지 트래킹

<pre>
<code>
python demo.py --model=../pretrained -vn data/NonVideo4 -lf 005
</pre>
</code>


명령행 옵션으로 --model, -vn, -lf를 인자값으로 받습니다.

--model은 pretrained폴더에 이미학습된 siamfc 모델

-vn은 읽어들일 plenoptic구조 데이터폴더

-lf는 pletoptic구조의 데이터에서 일반 이미지의 번호 입니다. (예: 모든 이미지의 images폴더의 005.png 파일들만 읽습니다.)  


---------------------------------------------------------------


## 2. vot_siamfc_3D_v0
### * focal 이미지 전부 읽기
프레임당 101장의 focal 이미지를 모두 읽어 1장씩 crop_and_resize를 합니다.

이 101장 중에 가장 높은 response 값을 갖는 scale_id에 해당하는 바운딩박스를 만들어
2D 이미지에 찍어 보여줍니다.

<pre>
<code>
python demo.py --model=../pretrained -vn ../data/NonVideo4 -lf 005
</pre>
</code>


명령행 옵션으로 --model -vn -lf를 인자값으로 받습니다.

--model은 pretrained폴더에 이미학습된 siamfc 모델

-vn은 읽어들일 plenoptic구조 데이터폴더

-lf는 pletoptic구조의 데이터에서 일반 이미지의 번호 입니다. (예: 모든 이미지의 images폴더의 005.png 파일들만 읽습니다.)  

---------------------------------------------------------------


## 3. vot_siamfc_3D_v1

<div>
<img width="800" src="https://user-images.githubusercontent.com/51473705/99151364-fe3d7300-26dd-11eb-9340-316e1f2b6e12.PNG">       
</div>

### * focal 선택적 방법 V1
프레임당 101장의 focal 이미지중 -lfs를 통해 직접 처음 focal 위치를 잡아줍니다.

처음 lfs를 기준으로 앞으로 1장 뒤로 1장(예: 26.png, 26.png, 27.png)씩 3장을 읽어 1장씩 crop_and_resize를 합니다.(5장씩으로 수정가능)

3장중 가장 높은 score를 보여주는 이미지를 선택하고 그 이미지 번호를 기준으로 다시 3장씩 읽어가는 방식으로 진행 됩니다.

<pre>
<code>
python demo.py --model=pretrained -vn ../data/NonVideo4 -lf 005 -lfs 27
</pre>
</code>

## 3. vot_siamfc_3D_v1_2

위 방법에서 직접 lfs를 정해주는것이 아니라 '선명도'를 통해 focal 위치를 잡아줍니다. 그 이후는 위 방법과 같이 진행됩니다.

<pre>
<code>
python demo.py --model=../pretrained -vn ../data/NonVideo4_1 -lf 005
</pre>
</code>

---------------------------------------------------------------


## 4. vot_siamfc_3D_v2

<div>
<img width="800" src="https://user-images.githubusercontent.com/51473705/99151382-1b724180-26de-11eb-9c4f-ed5377ae919c.PNG">       
</div>

### * focal 선택적 방법 V2
프레임당 101장의 focal 이미지중 -lfs를 통해 직접 처음 focal 위치를 잡아줍니다.

처음 lfs를 기준으로 앞으로 1장 뒤로 1장(예: 26.png, 26.png, 27.png)씩 3장을 읽어 3장씩 crop_and_resize를 합니다.(5장씩으로 수정가능)

crop된 이미지 9장중 가장 높은 score를 보여주는 이미지를 선택하고 그 이미지 번호를 기준으로 다시 3장씩 읽어가는 방식으로 진행 됩니다.

<pre>
<code>
python demo_mine.py --model=pretrained -vn ../data/NonVideo4 -lf 005 -lfs 27
</pre>
</code>

## 4. vot_siamfc_3D_v2_2
위 방법에서 직접 lfs를 정해주는것이 아니라 '선명도'를 통해 focal 위치를 잡아줍니다. 그 이후는 위 방법과 같이 진행됩니다.

<pre>
<code>
python demo.py --model=../pretrained -vn ../data/NonVideo4 -lf 005
</pre>
</code>

---------------------------------------------------------------

## 5. vot_siamfc_3D_v3

<div>
<img width="800" src="https://user-images.githubusercontent.com/51473705/99151393-2e851180-26de-11eb-9a46-d94fcabe99b7.PNG">       
</div>

### * focal 선택적 방법 V3
프레임당 101장의 focal 이미지중 -lfs를 통해 직접 처음 focal 위치를 잡아줍니다.

처음 lfs를 기준으로 앞으로 1장 뒤로 1장(예: 26.png, 26.png, 27.png)씩 3장을 읽어 3장씩 crop_and_resize를 합니다.(5장씩으로 수정가능)

crop된 이미지 각 3장중 가장 높은 score를 보여주는 이미지를 선택하여 각 3개의 respose를 뽑습니다. 그 3개중 다시 하나를 선택합니다. 

그 이미지 번호를 기준으로 다시 3장씩 읽어가는 방식으로 진행 됩니다.

<pre>
<code>
python demo.py --model=../pretrained -vn ../data/NonVideo4 -lf 005 -lfs 27
</pre>
</code>

## 5. vot_siamfc_3D_v3_2
위 방법에서 직접 lfs를 정해주는것이 아니라 선명도를 통해 focal 위치를 잡아줍니다. 그 이후는 위 방법과 같이 진행됩니다.

<pre>
<code>
python demo.py --model=../pretrained -vn ../data/NonVideo4_0 -lf 005
</pre>
</code>

---------------------------------------------------------------


각 모든 폴더에는 image2video.py 코드를 통해 트래킹하며 저장한 이미지를 동영상처럼 볼수 있습니다.

각 모든 폴더에는 read_groundtruth.py 코드를 통해 recode 폴더의 gt를 확인해 볼수 있습니다.

각 모든 폴더에는 test_groundtruth.py 코드를 통해 recode 폴더의 gt와 tracking 하면서 저장한 bbox의 위치 데이터를 읽어 서로 비교합니다.

gt와 bbox의 precision과 IOU를 구할수 있습니다.

siamfc_mine.py 에서 if visualize == True:

에 아래와 같은 코드를 추가하여 respose를 봄으로써 모델이 객체에서 어디를 중점으로 보는지 볼수 있다.

<pre>
<code>
#fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16,16))  #response.shape[0]
#fig.set_size_inches(12, 12)
#for i in range(3):
#    ax[i].imshow(search[i])
#    ax[i].imshow(response[i], cmap='jet', alpha=0.1)
#    ax[i].set_title('selected response[' + str(scale_id-1+i) +']')
#    ax[i].axis('off')  

#fig.tight_layout()
#save_plt = "./response/{0:0=3d}".format(f)
#plt.savefig(save_plt+'_'+str(scale_id)+'.png')

</pre>
</code>
