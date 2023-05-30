import numpy as np
from numpy import random
from math import asin
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan,CompressedImage
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from mobilenet import detect, net,CONFIDENCE, COLORS, CLASSES
import cv2.aruco as aruco

class Control():
    def __init__(self):
        self.rate = rospy.Rate(250)

        # HSV Filter
        self.color_param = {
            "green": {
                "lower": np.array([58, 255, 107], dtype=np.uint8),
                "upper": np.array([62, 255, 244] , dtype=np.uint8)  
            },
            "yellow": {
                "lower": np.array([(24, 214, 240)], dtype=np.uint8),
                "upper": np.array([(157, 245, 255)], dtype=np.uint8)
            },
            }
        self.kernel = np.ones((5,5),np.uint8)
		# Subscribers
        self.bridge = CvBridge()
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)
        self.laser_subscriber = rospy.Subscriber('/scan',LaserScan, self.laser_callback)
        self.image_sub = rospy.Subscriber('/camera/image/compressed', CompressedImage, self.image_callback, queue_size=1, buff_size = 2**24)
        #para a garra
        self.ombro = rospy.Publisher("/joint1_position_controller/command", Float64, queue_size=1)
        self.garra = rospy.Publisher("/joint2_position_controller/command", Float64, queue_size=1)	

        #camera 
        self.camera_distortion = np.loadtxt('/home/borg/catkin_ws/src/meu_projeto/scripts/cameraDistortion_realsense.txt', delimiter=',')
        self.camera_matrix = np.loadtxt('/home/borg/catkin_ws/src/meu_projeto/scripts/cameraMatrix_realsense.txt', delimiter=',')

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=3)



        self.cmd_vel_pub.publish(Twist())
        self.state = 1
        self.selected_mod = None
        self.robot_state = "anda"
        self.robot_machine = {
            "checar": self.checar,
            "center_on_coord": self.center_on_coord,
            "para": self.para,
            "anda": self.anda,
 
        }

        self.pista_machine = {
             "aproxima_pista" : self.aproxima_pista,
        }

        self.initial_position = 0
        self.n1 = random.randint(1,2) 
        self.kp = 200

    def odom_callback(self, data: Odometry):
        self.position = data.pose.pose.position

        if self.initial_position == 0:
            self.initial_position = self.position
        
        orientation_list = [data.pose.pose.orientation.x,
                            data.pose.pose.orientation.y,
                            data.pose.pose.orientation.z,
                            data.pose.pose.orientation.w]

        self.roll, self.pitch, self.yaw = euler_from_quaternion(orientation_list)

        self.yaw = self.yaw % (2*np.pi)
	
    def laser_callback(self, msg: LaserScan) -> None:
        self.laser_msg = np.array(msg.ranges).round(decimals=2)
        self.laser_msg[self.laser_msg == 0] = np.inf


        self.laser_forward = np.min(list(self.laser_msg[0:30]) + list(self.laser_msg[329:359]))
        self.laser_backwards = np.min(list(self.laser_msg[175:185]))


	

    def color_segmentation(self, hsv: np.ndarray, lower_hsv: np.ndarray, upper_hsv: np.ndarray,) -> Point:
        """ 
        Use HSV color space to segment the image and find the center of the object.

        Args:
            bgr (np.ndarray): image in BGR format
        
        Returns:
            Point: x, y and area of the object
        """
        point = []
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        contornos, arvore = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        maior_contorno = None
        maior_contorno_area = 0
        

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area > maior_contorno_area:
                maior_contorno = cnt
                maior_contorno_area = area

        M = cv2.moments(maior_contorno)

        if M["m00"] == 0:
            point_x = 0
            point_y = 0
            point = [point_x,point_y]

        else:
            point_x = int(M["m10"] / M["m00"])
            point_y = int(M["m01"] / M["m00"])
            point = [point_x,point_y]

        return mask,point, maior_contorno_area 

    def image_callback(self, msg: CompressedImage) -> None:
        """
        Callback function for the image topic
        """
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            img = cv_image.copy()
        except CvBridgeError as e:
            print(e)

        h, w, d = img.shape
        self.centro_segue = (h, 25*h//40) # corte que eu fiz só para que ele olhe para a pista
        self.centro_img = (w//2, h//2) # foi o corte que fiz pra ele olhar o centro da imagem integralmente 



        self.maska, self.yellow, self.area_y = self.color_segmentation(hsv, self.color_param["yellow"]["lower"], self.color_param["yellow"]["upper"])
        self.maskg, self.green, self.area_g = self.color_segmentation(hsv, self.color_param["green"]["lower"], self.color_param["green"]["upper"])
        


        if self.state == 1: 
            if self.yellow[0] !=0:
                self.selected_mod = "yellow"
                self.robot_state = "checar"

        # quiser mostrar algo da um um imgshow
        cv2.imshow("Referencia amarelo",self.maska)
        cv2.waitKey(1)

        cv2.imshow("Referencia verde",self.maskg)
        cv2.waitKey(1)

    def anda(self) -> None:
        self.twist = Twist()
        self.twist.linear.x = 0.03

    def checar(self) -> None:
        """
        Stop the robot
        """
        if self.selected_mod == "yellow":
            self.robot_machine.update(self.pista_machine)
            self.robot_state = "aproxima_pista"

        



    def aproxima_pista(self) -> None:
        self.center_on_coord()  #centraliza imagem com o centro da caixa      
        self.twist.linear.x = 0.2

        #quando a area que ele estiver vendo da caixa for igual a 1000 ele para 
        
        #se ele ta vendo a caixa
        if self.green[0] != 0:
            #verifica se a area da caixa é maior que um dado valor, vai testando até achar um valor que ele consiga ver a caixa e parar a uma distancia que vc queira
            #isso diz que se a area da caixa é maior o igual a 2000 ele para
            if self.area_g >= 20000:
                self.state = 2
                self.robot_state = "para"

    def para(self) -> None:
        self.twist = Twist() 
        if self.state == 2: 
            self.twist.angular.z = 0.0
            self.twist.linear.x = 0.0
            
            


    def center_on_coord(self):
        self.twist = Twist()
        err = 0
        
        if self.selected_mod == "yellow":
            err = self.centro_segue[0] - self.yellow[0]

        
        self.twist.angular.z = float(err) / self.kp # quanto maior o kp mais preciso vai ser o seu robo, quanto menor o kp menos preciso 

    def control(self):
        '''
        This function is called at least at {self.rate} Hz.
        This function controls the robot.
        '''
        self.twist = Twist()
        print(f'self.robot_state: {self.robot_state}')
        self.robot_machine[self.robot_state]()

        self.cmd_vel_pub.publish(self.twist)
        
        self.rate.sleep()

def main():
	rospy.init_node('Aleatorio')
	control = Control()
	rospy.sleep(1)

	while not rospy.is_shutdown():
		control.control()

if __name__=="__main__":
	main()