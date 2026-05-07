# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import argparse

from generate import generate, generate_with_prefix_cache, generate_with_dual_cache, generate_streaming_blocks
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM

def chat(args):
    device = 'cuda'
    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    gen_length = args.gen_length
    steps = args.steps
    print('*' * 66)
    print(f'**  Answer Length: {gen_length}  |  Sampling Steps: {steps}  **')
    print('*' * 66)

    conversation_num = 0
    while True:
        user_input = \
        '''
        You are a world-class math competition competitor

Here is a problem and a few sample solutions:
	
	
Let $x,$ $y,$ and $z$ be positive real numbers satisfying the system of equations:
\begin{align*} \sqrt{2x-xy} + \sqrt{2y-xy} &= 1 \\ \sqrt{2y-yz} + \sqrt{2z-yz} &= \sqrt2 \\ \sqrt{2z-zx} + \sqrt{2x-zx} &= \sqrt3. \end{align*} 
Then $\left[ (1-x)(1-y)(1-z) \right]^2$ can be written as $\frac{m}{n},$ where $m$ and $n$ are relatively prime positive integers. Find $m+n.$
First, let define a triangle with side lengths $\sqrt{2x}$, $\sqrt{2z}$, and $l$, with altitude from $l$'s equal to $\sqrt{xz}$. $l = \sqrt{2x - xz} + \sqrt{2z - xz}$, the left side of one equation in the problem.
Let  $\theta$ be angle opposite the side with length $\sqrt{2x}$. Then the altitude has length $\sqrt{2z} \cdot \sin(\theta) = \sqrt{xz}$ and thus $\sin(\theta) = \sqrt{\frac{x}{2}}$, so $x=2\sin^2(\theta)$ and the side length $\sqrt{2x}$ is equal to $2\sin(\theta)$.
We can symmetrically apply this to the two other equations/triangles. 
By law of sines, we have $\frac{2\sin(\theta)}{\sin(\theta)} = 2R$, with $R=1$ as the circumradius, same for all 3 triangles. 
The circumcircle's central angle to a side is $2 \arcsin(l/2)$, so the 3 triangles' $l=1, \sqrt{2}, \sqrt{3}$,  have angles $120^{\circ}, 90^{\circ}, 60^{\circ}$, respectively.
This means that by half angle arcs, we see that we have in some order, $x=2\sin^2(\alpha)$, $y=2\sin^2(\beta)$, and $z=2\sin^2(\gamma)$ (not necessarily this order, but here it does not matter due to symmetry), satisfying that $\alpha+\beta=180^{\circ}-\frac{120^{\circ}}{2}$, $\beta+\gamma=180^{\circ}-\frac{90^{\circ}}{2}$, and $\gamma+\alpha=180^{\circ}-\frac{60^{\circ}}{2}$. Solving, we get $\alpha=\frac{135^{\circ}}{2}$, $\beta=\frac{105^{\circ}}{2}$, and $\gamma=\frac{165^{\circ}}{2}$.
We notice that \[[(1-x)(1-y)(1-z)]^2=[\sin(2\alpha)\sin(2\beta)\sin(2\gamma)]^2=[\sin(135^{\circ})\sin(105^{\circ})\sin(165^{\circ})]^2\] \[=\left(\frac{\sqrt{2}}{2} \cdot \frac{\sqrt{6}-\sqrt{2}}{4} \cdot \frac{\sqrt{6}+\sqrt{2}}{4}\right)^2 = \left(\frac{\sqrt{2}}{8}\right)^2=\frac{1}{32} \to \boxed{033}. \blacksquare\]
- kevinmathz
(This eventually whittles down to the same concept as Solution 1)
Note that in each equation in this system, it is possible to factor $\sqrt{x}$, $\sqrt{y}$, or $\sqrt{z}$ from each term (on the left sides), since each of $x$, $y$, and $z$ are positive real numbers. After factoring out accordingly from each terms one of $\sqrt{x}$, $\sqrt{y}$, or $\sqrt{z}$, the system should look like this:
\begin{align*} \sqrt{x}\cdot\sqrt{2-y} + \sqrt{y}\cdot\sqrt{2-x} &= 1 \\ \sqrt{y}\cdot\sqrt{2-z} + \sqrt{z}\cdot\sqrt{2-y} &= \sqrt2 \\ \sqrt{z}\cdot\sqrt{2-x} + \sqrt{x}\cdot\sqrt{2-z} &= \sqrt3. \end{align*}
This should give off tons of trigonometry vibes. To make the connection clear, $x = 2\cos^2 \alpha$, $y = 2\cos^2 \beta$, and $z = 2\cos^2 \theta$ is a helpful substitution:
\begin{align*} \sqrt{2\cos^2 \alpha}\cdot\sqrt{2-2\cos^2 \beta} + \sqrt{2\cos^2 \beta}\cdot\sqrt{2-2\cos^2 \alpha} &= 1 \\ \sqrt{2\cos^2 \beta}\cdot\sqrt{2-2\cos^2 \theta} + \sqrt{2\cos^2 \theta}\cdot\sqrt{2-2\cos^2 \beta} &= \sqrt2 \\ \sqrt{2\cos^2 \theta}\cdot\sqrt{2-2\cos^2 \alpha} + \sqrt{2\cos^2 \alpha}\cdot\sqrt{2-2\cos^2 \theta} &= \sqrt3. \end{align*}
From each equation $\sqrt{2}^2$ can be factored out, and when every equation is divided by 2, we get:
\begin{align*} \sqrt{\cos^2 \alpha}\cdot\sqrt{1-\cos^2 \beta} + \sqrt{\cos^2 \beta}\cdot\sqrt{1-\cos^2 \alpha} &= \frac{1}{2} \\ \sqrt{\cos^2 \beta}\cdot\sqrt{1-\cos^2 \theta} + \sqrt{\cos^2 \theta}\cdot\sqrt{1-\cos^2 \beta} &= \frac{\sqrt2}{2} \\ \sqrt{\cos^2 \theta}\cdot\sqrt{1-\cos^2 \alpha} + \sqrt{\cos^2 \alpha}\cdot\sqrt{1-\cos^2 \theta} &= \frac{\sqrt3}{2}. \end{align*}
which simplifies to (using the Pythagorean identity $\sin^2 \phi + \cos^2 \phi = 1 \; \forall \; \phi \in \mathbb{C}$):
\begin{align*} \cos \alpha\cdot\sin \beta + \cos \beta\cdot\sin \alpha &= \frac{1}{2} \\ \cos \beta\cdot\sin \theta + \cos \theta\cdot\sin \beta &= \frac{\sqrt2}{2} \\ \cos \theta\cdot\sin \alpha + \cos \alpha\cdot\sin \theta &= \frac{\sqrt3}{2}. \end{align*}
which further simplifies to (using sine addition formula $\sin(a + b) = \sin a \cos b + \cos a \sin b$):
\begin{align*} \sin(\alpha + \beta) &= \frac{1}{2} \\ \sin(\beta + \theta) &= \frac{\sqrt2}{2} \\ \sin(\alpha + \theta) &= \frac{\sqrt3}{2}. \end{align*}
Taking the inverse sine ($0\leq\theta\frac{\pi}{2}$) of each equation yields a simple system:
\begin{align*} \alpha + \beta &= \frac{\pi}{6} \\ \beta + \theta &= \frac{\pi}{4} \\ \alpha + \theta &= \frac{\pi}{3} \end{align*}
giving solutions:
\begin{align*} \alpha &= \frac{\pi}{8} \\ \beta &= \frac{\pi}{24} \\ \theta &= \frac{5\pi}{24} \end{align*}
Since these unknowns are directly related to our original unknowns, there are consequent solutions for those: 

\begin{align*} x &= 2\cos^2\left(\frac{\pi}{8}\right) \\ y &= 2\cos^2\left(\frac{\pi}{24}\right) \\ z &= 2\cos^2\left(\frac{5\pi}{24}\right) \end{align*}
When plugging into the expression $\left[ (1-x)(1-y)(1-z) \right]^2$, noting that $-\cos 2\phi = 1 - 2\cos^2 \phi\; \forall \; \phi \in \mathbb{C}$ helps to simplify this expression into:

\begin{align*} \left[ (-1)^3\left(\cos \left(2\cdot\frac{\pi}{8}\right)\cos \left(2\cdot\frac{\pi}{24}\right)\cos \left(2\cdot\frac{5\pi}{24}\right)\right)\right]^2 \\ = \left[ (-1)\left(\cos \left(\frac{\pi}{4}\right)\cos \left(\frac{\pi}{12}\right)\cos \left(\frac{5\pi}{12}\right)\right)\right]^2 \end{align*}
Now, all the cosines in here are fairly standard: 
\begin{align*} \cos \frac{\pi}{4} &= \frac{\sqrt{2}}{2} \\ \cos \frac{\pi}{12} &=\frac{\sqrt{6} + \sqrt{2}}{4} & (= \cos{\frac{\frac{\pi}{6}}{2}} ) \\ \cos \frac{5\pi}{12} &= \frac{\sqrt{6} - \sqrt{2}}{4} & (=  \cos\left({\frac{\pi}{6} + \frac{\pi}{4}} \right) ) \end{align*}
With some final calculations:

\begin{align*} &(-1)^2\left(\frac{\sqrt{2}}{2}\right)^2\left(\frac{\sqrt{6} + \sqrt{2}}{4}\right)^2\left(\frac{\sqrt{6} - \sqrt{2}}{4}\right)^2 \\ =& \left(\frac{1}{2}\right) \left(\left(\frac{\sqrt{6} + \sqrt{2}}{4}\right)\left(\frac{\sqrt{6} - \sqrt{2}}{4}\right)\right)^2 \\ =&\frac{1}{2} \frac{4^2}{16^2} = \frac{1}{32} \end{align*}
This is our answer in simplest form $\frac{m}{n}$, so $m + n = 1 + 32 = \boxed{033}$.
~Oxymoronic15
Let $1-x=a;1-y=b;1-z=c$, rewrite those equations
$\sqrt{(1-a)(1+b)}+\sqrt{(1+a)(1-b)}=1$;
$\sqrt{(1-b)(1+c)}+\sqrt{(1+b)(1-c)}=\sqrt{2}$
$\sqrt{(1-a)(1+c)}+\sqrt{(1-c)(1+a)}=\sqrt{3}$
and solve for $m/n = (abc)^2 = a^2b^2c^2$
Square both sides and simplify, to get three equations:
$2ab-1=2\sqrt{(1-a^2)(1-b^2)}$
$2bc~ ~ ~ ~  ~ ~=2\sqrt{(1-b^2)(1-c^2)}$
$2ac+1=2\sqrt{(1-c^2)(1-a^2)}$
Square both sides again, and simplify to get three equations:
$a^2+b^2-ab=\frac{3}{4}$
$b^2+c^2~ ~ ~ ~ ~ ~=1$
$a^2+c^2+ac=\frac{3}{4}$
Subtract first and third equation, getting $(b+c)(b-c)=a(b+c)$, $a=b-c$
Put it in first equation, getting $b^2-2bc+c^2+b^2-b(b-c)=b^2+c^2-bc=\frac{3}{4}$, $bc=\frac{1}{4}$
Since $a^2=b^2+c^2-2bc=\frac{1}{2}$, $m/n = a^2b^2c^2 = a^2(bc)^2 = \frac{1}{2}\left(\frac{1}{4}\right)^2=\frac{1}{32}$  and so the final answer is $\boxed{033}$
~bluesoul
Denote $u = 1 - x$, $v = 1 - y$, $w = 1 - z$.
Hence, the system of equations given in the problem can be written as
\begin{align*} \sqrt{(1-u)(1+v)} + \sqrt{(1+u)(1-v)} & = 1 \hspace{1cm} (1) \\ \sqrt{(1-v)(1+w)} + \sqrt{(1+v)(1-w)} & = \sqrt{2} \hspace{1cm} (2) \\ \sqrt{(1-w)(1+u)} + \sqrt{(1+w)(1-u)} & = \sqrt{3} . \hspace{1cm} (3)  \end{align*}
Each equation above takes the following form:
\[ \sqrt{(1-a)(1+b)} + \sqrt{(1+a)(1-b)} = k . \]
Now, we simplify this equation by removing radicals.
Denote $p = \sqrt{(1-a)(1+b)}$ and $q = \sqrt{(1+a)(1-b)}$.
Hence, the equation above implies
\[ \left\{ \begin{array}{l} p + q = k \\ p^2 = (1-a)(1+b) \\ q^2 = (1+a)(1-b) \end{array} \right.. \]
Hence, $q^2 - p^2 = (1+a)(1-b) - (1-a)(1+b) = 2 (a-b)$.
Hence, $q - p = \frac{q^2 - p^2}{p+q} = \frac{2}{k} (a-b)$.
Because $p + q = k$ and $q - p = \frac{2}{k} (a-b)$, we get $q = \frac{a-b}{k} + \frac{k}{2}$.
Plugging this into the equation $q^2 = (1+a)(1-b)$ and simplifying it, we get
\[ a^2 + \left( k^2 - 2 \right) ab + b^2 = k^2 - \frac{k^4}{4} . \]
Therefore, the system of equations above can be simplified as
\begin{align*} u^2 - uv + v^2 & = \frac{3}{4} \\ v^2 + w^2 & = 1 \\ w^2 + wu + u^2 & = \frac{3}{4}  . \end{align*}
Denote $w' = - w$.
The system of equations above can be equivalently written as
\begin{align*} u^2 - uv + v^2 & = \frac{3}{4} \hspace{1cm} (1') \\ v^2 + w'^2 & = 1 \hspace{1cm} (2') \\ w'^2 - w'u + u^2 & = \frac{3}{4} \hspace{1cm} (3') . \end{align*}
Taking $(1') - (3')$, we get
\[ (v - w') (v + w' - u) = 0 . \]
Thus, we have either $v - w' = 0$ or $v + w' - u = 0$.
$\textbf{Case 1}$: $v - w' = 0$.
Equation (2') implies $v = w' = \pm \frac{1}{\sqrt{2}}$.
Plugging $v$ and $w'$ into Equation (2), we get contradiction. Therefore, this case is infeasible.
$\textbf{Case 2}$: $v + w' - u = 0$.
Plugging this condition into (1') to substitute $u$, we get
\[ v^2 + v w' + w'^2 = \frac{3}{4} \hspace{1cm} (4) . \]
Taking $(4) - (2')$, we get
\[ v w' = - \frac{1}{4} . \hspace{1cm} (5) . \]
Taking (4) + (5), we get
\[ \left( v + w' \right)^2 = \frac{1}{2} . \]
Hence, $u^2 = \left( v + w' \right)^2 = \frac{1}{2}$.
Therefore,
\begin{align*} \left[ (1-x)(1-y)(1-z) \right]^2 & = u^2 (vw)^2 \\ & = u^2 (vw')^2 \\ & = \frac{1}{2} \left( - \frac{1}{4} \right)^2 \\ & = \frac{1}{32} . \end{align*}
Therefore, the answer is $1 + 32 = \boxed{\textbf{(033) }}$.
~Steven Chen (www.professorchenedu.com)
bu-bye
\begin{align*} \sin(\alpha + \beta) &= 1/2 \\  \sin(\alpha + \gamma) &= \sqrt2/2 \\  \sin(\beta + \gamma) &= \sqrt3/2. \end{align*}
Thus, 
\begin{align*} \alpha + \beta &= 30^{\circ} \\  \alpha + \gamma &= 45^{\circ} \\  \beta + \gamma &= 60^{\circ}, \end{align*}
so $(\alpha, \beta, \gamma) = (15/2^{\circ}, 45/2^{\circ}, 75/2^{\circ})$. Hence,
\[abc = (1-2\sin^2(\alpha))(1-2\sin^2(\beta))(1-2\sin^2(\gamma))=\cos(15^{\circ})\cos(45^{\circ})\cos(75^{\circ})=\frac{\sqrt{2}}{8},\]
so $(abc)^2=(\sqrt{2}/8)^2=\frac{1}{32}$, for a final answer of $\boxed{033}$.
Remark
The motivation for the trig substitution is that if $\sin^2(\alpha)=(1-a)/2$, then $\cos^2(\alpha)=(1+a)/2$, and when making the substitution in each equation of the initial set of equations, we obtain a new equation in the form of the sine addition formula.
~ Leo.Euler
[2022 AIME I 15.png](https://artofproblemsolving.com/wiki/index.php/File:2022_AIME_I_15.png)
In given equations, $0 \leq x,y,z \leq 2,$ so we define some points:
\[\bar {O} = (0, 0), \bar {A} = (1, 0), \bar{M} = \left(\frac {1}{\sqrt{2}},\frac {1}{\sqrt{2}}\right),\]
\[\bar {X} = \left(\sqrt {\frac {x}{2}}, \sqrt{1 – \frac{x}{2}}\right), \bar {Y'} = \left(\sqrt {\frac {y}{2}}, \sqrt{1 – \frac{y}{2}}\right),\]
\[\bar {Y} = \left(\sqrt {1 – \frac{y}{2}},\sqrt{\frac {y}{2}}\right), \bar {Z} = \left(\sqrt {1 – \frac{z}{2}},\sqrt{\frac {z}{2}}\right).\]
Notice, that \[\mid \vec {AO} \mid = \mid \vec {MO} \mid = \mid \vec {XO} \mid =\mid \vec {YO} \mid = \mid \vec {Y'O} \mid =\mid \vec {ZO} \mid = 1\] and each points lies in the first quadrant.
We use given equations and get some scalar products:
\[(\vec {XO} \cdot \vec {YO}) = \frac {1}{2} = \cos \angle XOY \implies  \angle XOY = 60 ^\circ,\]
\[(\vec {XO} \cdot \vec {ZO}) = \frac {\sqrt{3}}{2} = \cos \angle XOZ \implies  \angle XOZ = 30^\circ,\]
\[(\vec {Y'O} \cdot \vec {ZO}) = \frac {1}{\sqrt{2}} = \cos \angle Y'OZ \implies  \angle Y'OZ = 45^\circ.\] 
So $\angle YOZ =  \angle XOY – \angle XOZ =  60 ^\circ – 30 ^\circ = 30 ^\circ,   \angle Y'OY =  \angle Y'OZ +  \angle YOZ = 45^\circ + 30 ^\circ = 75^\circ.$
Points $Y$ and $Y'$ are symmetric with respect to $OM.$
Case 1
\[\angle YOA = \frac{90^\circ – 75^\circ}{2} = 7.5^\circ, \angle ZOA = 30^\circ + 7.5^\circ = 37.5^\circ, \angle XOA = 60^\circ + 7.5^\circ = 67.5^\circ .\]
\[1 – x = \left(\sqrt{1 – \frac{x}{2}} \right)^2– \left(\sqrt{\frac {x}{2}}\right)^2  = \sin^2 \angle XOA – \cos^2 \angle XOA = –\cos 2 \angle XOA = –\cos 135^\circ,\]
\[1 – y = \cos 15^\circ, 1 – z = \cos 75^\circ \implies  \left[ (1–x)(1–y)(1–z) \right]^2 = \left[ \sin 45^\circ \cdot \cos 15^\circ \cdot \sin 15^\circ \right]^2 =\]
\[=\left[ \frac {\sin 45^\circ \cdot \sin 30^\circ}{2} \right]^2  = \frac {1}{32} \implies \boxed{\textbf{033}}.\]
Case 2
\[\angle Y_1 OA = \frac{90^\circ + 75^\circ}{2} = 82.5^\circ, \angle Z_1 OA = 82.5^\circ – 30^\circ = 52.5^\circ, \angle X_1 OA = 82.5^\circ – 60^\circ = 22.5^\circ \implies \boxed{\textbf{033}}.\]
vladimir.shelomovskii@gmail.com, vvsss

Now solve the following problem, providing a similar amount of detail in your solution:

 	
For any finite set $X$, let $| X |$ denote the number of elements in $X$. Define
\[S_n = \sum | A \cap B | ,\]
where the sum is taken over all ordered pairs $(A, B)$ such that $A$ and $B$ are subsets of $\left\{ 1 , 2 , 3,  \cdots , n \right\}$ with $|A| = |B|$.
For example, $S_2 = 4$ because the sum is taken over the pairs of subsets
\[(A, B) \in \left\{ (\emptyset, \emptyset) , ( \{1\} , \{1\} ), ( \{1\} , \{2\} ) , ( \{2\} , \{1\} ) , ( \{2\} , \{2\} ) , ( \{1 , 2\} , \{1 , 2\} ) \right\} ,\]
giving $S_2 = 0 + 1 + 0 + 0 + 1 + 2 = 4$.
Let $\frac{S_{2022}}{S_{2021}} = \frac{p}{q}$, where $p$ and $q$ are relatively prime positive integers. Find the remainder when $p + q$ is divided by
1000.



        '''

        m = [{"role": "user", "content": user_input}]
        user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(user_input)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        if conversation_num == 0:
            prompt = input_ids
        else:
            prompt = torch.cat([prompt, input_ids[:, 1:]], dim=1)
        print(f'use cache: {args.use_cache} use cache position: {args.if_cache_position} threshold: {args.threshold} block size: {args.block_size} streaming: {args.streaming}')
        if args.streaming:
            out, nfe = generate_streaming_blocks(
                model, prompt,
                steps=steps,
                block_length=args.block_size,
                max_gen_length=gen_length,
                temperature=0.,
                remasking='low_confidence',
                eos_id=tokenizer.eos_token_id,
                threshold=args.threshold,
                use_cache=args.use_cache,
            )
        elif args.use_cache:
            if args.if_cache_position:
                out, nfe = generate_with_dual_cache(model, prompt, steps=steps, gen_length=gen_length, block_length=args.block_size, temperature=0., remasking='low_confidence', threshold=args.threshold)
            else:
                out, nfe = generate_with_prefix_cache(model, prompt, steps=steps, gen_length=gen_length, block_length=args.block_size, temperature=0., remasking='low_confidence', threshold=args.threshold)
        else:
            out, nfe = generate(model, prompt, steps=steps, gen_length=gen_length, block_length=args.block_size, temperature=0., remasking='low_confidence', threshold=args.threshold)

        answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
        print(f"Bot's reply: {answer}")
        print(f"Number of forward passes: {nfe}")

        # remove the <EOS>
        prompt = out[out != 126081].unsqueeze(0)
        conversation_num += 1
        print('-----------------------------------------------------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--if_cache_position", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--streaming", action="store_true",
                        help="Decode one block at a time with no suffix masks. "
                             "--steps is interpreted as steps-per-block. "
                             "--use_cache enables prefix KV-cache for each block.")

    args = parser.parse_args()
    chat(args)

