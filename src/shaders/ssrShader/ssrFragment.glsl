#ifdef GL_ES
precision highp float;
#endif

uniform vec3 uLightDir;
uniform vec3 uCameraPos;
uniform vec3 uLightRadiance;
uniform sampler2D uGDiffuse;
uniform sampler2D uGDepth;
uniform sampler2D uGNormalWorld;
uniform sampler2D uGShadow;
uniform sampler2D uGPosWorld;

varying mat4 vWorldToScreen;
varying highp vec4 vPosWorld;

#define M_PI 3.1415926535897932384626433832795
#define TWO_PI 6.283185307
#define INV_PI 0.31830988618
#define INV_TWO_PI 0.15915494309

float Rand1(inout float p) {
  p = fract(p * .1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}

vec2 Rand2(inout float p) {
  return vec2(Rand1(p), Rand1(p));
}

float InitRand(vec2 uv) {
	vec3 p3  = fract(vec3(uv.xyx) * .1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

vec3 SampleHemisphereUniform(inout float s, out float pdf) {//均匀采样，半球的pdf就是2pi分之一
  vec2 uv = Rand2(s);
  float z = uv.x;//随机数[0,1]作为z的范围
  float phi = uv.y * TWO_PI;//随机数[0,1]再乘以2PI，范围是[0,2PI]就是方位角
  //上面就相当于随机取了z和方位角
  float sinTheta = sqrt(1.0 - z*z);//xy对角线长度
  vec3 dir = vec3(sinTheta * cos(phi), sinTheta * sin(phi), z);//笛卡尔坐标系xyz
  pdf = INV_TWO_PI;
  return dir;
}

vec3 SampleHemisphereCos(inout float s, out float pdf) {
  vec2 uv = Rand2(s);
  float z = sqrt(1.0 - uv.x);
  float phi = uv.y * TWO_PI;
  float sinTheta = sqrt(uv.x);
  vec3 dir = vec3(sinTheta * cos(phi), sinTheta * sin(phi), z);
  pdf = z * INV_PI;
  return dir;
}

//通过传入WorldSpace下的法线n，返回两个切线向量。
//可以通过他们将随机采样得到的方向变换到世界坐标系中
void LocalBasis(vec3 n, out vec3 b1, out vec3 b2) {
  float sign_ = sign(n.z);
  if (n.z == 0.0) {
    sign_ = 1.0;
  }
  float a = -1.0 / (sign_ + n.z);
  float b = n.x * n.y * a;
  b1 = vec3(1.0 + sign_ * n.x * n.x * a, sign_ * b, -sign_ * n.x);
  b2 = vec3(b, sign_ + n.y * n.y * a, -n.y);
}

vec4 Project(vec4 a) {
  return a / a.w;
}

float GetDepth(vec3 posWorld) {
  float depth = (vWorldToScreen * vec4(posWorld, 1.0)).w;
  return depth;
}

/*
 * Transform point from world space to screen space([0, 1] x [0, 1])
 *
 */
vec2 GetScreenCoordinate(vec3 posWorld) {
  vec2 uv = Project(vWorldToScreen * vec4(posWorld, 1.0)).xy * 0.5 + 0.5;
  return uv;
}

float GetGBufferDepth(vec2 uv) {
  float depth = texture2D(uGDepth, uv).x;//uGDepth存的也是clipSpace下w的值，即viewSpace下-Zeye的值
  if (depth < 1e-2) {
    depth = 1000.0;
  }
  return depth;
}

vec3 GetGBufferNormalWorld(vec2 uv) {
  vec3 normal = texture2D(uGNormalWorld, uv).xyz;
  return normal;
}

vec3 GetGBufferPosWorld(vec2 uv) {
  vec3 posWorld = texture2D(uGPosWorld, uv).xyz;
  return posWorld;
}

float GetGBufferuShadow(vec2 uv) {
  float visibility = texture2D(uGShadow, uv).x;//uGShadow就是一张GBuffer里面存的值是1或者0，已经在gbufferFragment的SimpleShadowMap方法算好了可见性
  return visibility;
}

vec3 GetGBufferDiffuse(vec2 uv) {
  vec3 diffuse = texture2D(uGDiffuse, uv).xyz;
  diffuse = pow(diffuse, vec3(2.2));
  return diffuse;
}

/*
 * Evaluate diffuse bsdf value.
 *
 * wi, wo are all in world space.
 * uv is in screen space, [0, 1] x [0, 1].
 *
 */
vec3 EvalDiffuse(vec3 wi, vec3 wo, vec2 uv) {
  vec3 L = vec3(0.0);
  
  vec3 albedo = GetGBufferDiffuse(uv);//获取反射率
  vec3 worldNormal = GetGBufferNormalWorld(uv);
  float cosTerm = max(0.0,dot(normalize(wi),worldNormal));
  L = albedo * INV_PI * cosTerm; // albedo除以PI 再乘上入射角的cos值

  return L;
}

/*
 * Evaluate directional light with shadow map
 * uv is in screen space, [0, 1] x [0, 1].
 *
 */
vec3 EvalDirectionalLight(vec2 uv) {//返回着色点受到光照的辐射度，考虑是否在阴影中
  vec3 Le = vec3(0.0);

  float visibility = GetGBufferuShadow(uv);//返回非0即1,判断这个uv点是否受直接光照射
  Le = uLightRadiance * visibility;
  
  return Le;
}

//返回是否相交，若相交再返回hitPos就是交点
bool RayMarch(vec3 ori, vec3 dir, out vec3 hitPos) {
  float step =0.05;//这里采用逐步判断法，走40步，没有采用课程中的mipmap方法
  vec3 endPoint = ori;
  for(int i=0; i<40; i++)
  {
    vec3 curPos = endPoint +step * dir;
    float curDepth = GetDepth(curPos);//就是取这个点在EyeSpace下的z值
    float bufferDepth = GetGBufferDepth(GetScreenCoordinate(curPos));//摄像机DepthBuffer存的z值
    //因为存的都是ViewSpace下-Zeye的值，离的越远-Zeye值越大
    if(step>40.0)
      return false;
    else if(curDepth > bufferDepth + 1e-6)
    {
      hitPos = curPos;
      return true;
    }
    else
    {
      endPoint = curPos;
    }
  }
  return false;
}

#define SAMPLE_NUM 1

void main() {
  float s = InitRand(gl_FragCoord.xy);

  vec3 L = vec3(0.0);
  //L = GetGBufferDiffuse(GetScreenCoordinate(vPosWorld.xyz));
  
  vec2 uv = GetScreenCoordinate(vPosWorld.xyz);
  vec3 wi = uLightDir;
  vec3 wo = uCameraPos - vPosWorld.xyz;
  L = EvalDiffuse(wi,wo,uv)*EvalDirectionalLight(uv);//直接光照

  vec3 L_indirect = vec3(0.0);//在蒙特卡洛中表示求和式里面的值，即一次Li*BRDF*cos/pdf
  for(int i=0; i<SAMPLE_NUM; i++)
  {
    Rand1(s);//把s变为[0,1]的随机数
    float pdf;
    vec3 dir = SampleHemisphereUniform(s,pdf);//返回一个局部坐标系  返回一个半球上随机的方向

    vec3 worldNormal = normalize(GetGBufferNormalWorld(uv));
    vec3 b1,b2;
    LocalBasis(worldNormal,b1,b2);//根据法线构建局部坐标系

    dir = normalize(mat3(b1,b2,worldNormal) * dir);//将随机得到的半球上的方向，变换到这个Normal所在局部坐标系下的世界方向

    vec3 hitPos;
    if(RayMarch(vPosWorld.xyz, dir, hitPos))//从这个着色点出发，随机发射一根方向为dir的光线，如果打到返回true和交点hitPos
    {
      vec2 uvQ = GetScreenCoordinate(hitPos);//交点q点的屏幕空间uv值，好计算它的radiance是多少
      vec3 shadeQ = EvalDiffuse(wi,-dir,uvQ)*EvalDirectionalLight(uvQ);//假设q点向p点发出的radiance和人眼看到的radiance是相同的，即总是假设次级光源是diffuse的
      L_indirect +=shadeQ * EvalDiffuse(dir,wo,uv) / pdf;  //从dir方向来的次级光源，照亮p点的BRDF值
    }
    L_indirect = L_indirect/float(SAMPLE_NUM);
    L+=L_indirect;
  }
  
  vec3 color = pow(clamp(L, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));
  gl_FragColor = vec4(vec3(color.rgb), 1.0);
}
