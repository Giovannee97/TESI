module QCLP
    implicit none

    public :: norm,omega_xy,obfun,omega_pol,sa

    contains

    subroutine omega_xy(x,y,env_prob,obs_prob,rew,gamma,res)
        real(8), intent(in),dimension(:,:,:,:)                          :: x
        real(8), intent(in),dimension(:,:,:)                            :: env_prob,obs_prob
        real(8), intent(in),dimension(:,:)                              :: y,rew
        real(8), intent(in)                                             :: gamma
        integer                                                         :: dimS,dimA,dimQ,dimO
        real(8), intent(out), dimension(size(x,1),size(env_prob,1))     :: res
        real(8), dimension(size(x,1),size(env_prob,1))                  :: term1,term2
        integer                                                         :: i,j,k,l,m,z

        term1=0.0
        term2=0.0

        dimS=size(env_prob,1)
        dimA=size(x,2)
        dimQ=size(x,1)
        dimO=size(x,4)
        
        !state index s
        do i=1,dimS
            !node index q
            do j=1,dimQ
                !nodeprime index q'
                do k=1,dimQ
                    !action index a
                    do l=1,dimA
                        !Calculation of reward dependent term with fixed observation
                        term1(j,i)=term1(j,i)+x(k,l,j,1)*rew(i,l)
                    enddo
                enddo
            enddo
        enddo

        !state index s
        do i=1,dimS
            !node index q
            do j=1,dimQ
                !nodeprime index q'
                do k=1,dimQ
                    !action index a
                    do l=1,dimA
                        !stateprime index s'
                        do m=1,dimS
                            !observation index o
                            do z=1,dimO
                                term2(j,i)=term2(j,i)+x(k,l,j,z)*env_prob(m,i,l)*obs_prob(z,m,l)*y(k,m)
                            enddo
                        enddo
                    enddo
                enddo
            enddo
        enddo

        res=term1+gamma*term2
    end 

    subroutine omega_pol(x,res)
        real(8), intent(in), dimension(:,:,:,:)                         :: x
        integer                                                         :: i,j,k,l,picker
        real(8)                                                         :: s
        real(8), dimension(size(x,2),size(x,3),size(x,4))               :: sum1,sum2
        real(8), intent(out), dimension(size(x,2),size(x,3),size(x,4))  :: res
    
        call random_number(s)
        picker=int(s*size(x,4))+1

        sum1=0.0
        sum2=0.0

        !index on a
        do j=1,size(x,2)
            !index on q
            do k=1,size(x,3)
                !index on o
                do l=1,size(x,4)
                    !index on q'
                    do i=1,size(x,1)
                        sum1(j,k,l)=x(i,j,k,l)+sum1(j,k,l)
                        sum2(j,k,l)=x(i,j,k,picker)+sum2(j,k,l)
                    enddo
                enddo
            enddo
        enddo

        res=sum1-sum2
    end subroutine
    
    subroutine norm(x,res)
        real(8), intent(in), dimension(:,:,:,:)                                     :: x
        integer                                                                     :: dimA,dimQ,dimO
        integer                                                                     :: i,j,k,l
        real(8), dimension(size(x,1),size(x,4))                                     :: norma
        real(8), intent(out), dimension(size(x,1),size(x,2),size(x,3),size(x,4))    :: res

        res=0.0
        norma=0.0

        dimA=size(x,2)
        dimQ=size(x,1)
        dimO=size(x,4)

        do i=1,dimQ
            do j=1,dimO
                do k=1,dimQ
                    do l=1,dimA
                        norma(i,j)=norma(i,j)+x(k,l,i,j)
                    enddo
                enddo
            enddo
        enddo

        do i=1,dimQ
            do j=1,dimO
                res(:,:,i,j)=x(:,:,i,j)/norma(i,j)
            enddo
        enddo

    end subroutine

    subroutine obfun(x,y,env_prob,obs_prob,rew,sigma,mu,gamma,val)
        real(8), intent(in),dimension(:,:,:,:)              :: x
        real(8), intent(in),dimension(:,:,:)                :: env_prob,obs_prob
        real(8), intent(in),dimension(:,:)                  :: y,rew
        real(8), intent(in)                                 :: sigma,gamma,mu
        real(8), dimension(size(x,1),size(env_prob,1))      :: om
        real(8), dimension(size(x,2),size(x,3),size(x,4))   :: om_pol
        real(8), dimension(size(env_prob,1))                :: init_prob
        real(8)                                             :: term1,term2,term3
        real(8), intent(out)                                :: val

        init_prob=1.0/size(env_prob,1)

        term1=0.0
        term2=0.0
        term3=0.0

        !First term of the objective function with fixed inital node qo=1
        term1=-sum(init_prob*y(1,:))

        !Penalty term for Bellman constraint
        call omega_xy(x,y,env_prob,obs_prob,rew,gamma,om)
        term2=sigma*sum((y-om)**2)

        !Penalty term for independence of policy from observation
        call omega_pol(x,om_pol)
        term3=mu*sum(om_pol**2)

        val=term1+term2+term3

        if (sigma > 100.0) then
            if (mod(int(sigma/100),2)==0) then 
                write(333,*) sigma,term1,term2,term3
            endif
        endif

    end subroutine

    subroutine sa(T,Tth,factor,n,x_init,y_init,rew,env_prob,obs_prob,sigma,mu,rndx,rndy,gamma,x_min,y_min,val_min)
        real(8), intent(in)                                                                             :: T,Tth,factor
        real(8), intent(in)                                                                             :: sigma,gamma,mu
        real(8), intent(in), dimension(:,:,:,:)                                                         :: rndx, x_init
        real(8), intent(in), dimension(:,:,:)                                                           :: env_prob,obs_prob
        real(8), intent(in), dimension(:,:)                                                             :: rew,rndy,y_init
        integer                                                                                         :: dimS,dimO,n
        integer                                                                                         :: i,countx,county
        real(8)                                                                                         :: T_inst,s,val_new
        real(8)                                                                                         :: val_old
        real(8), dimension(size(x_init,1),size(x_init,2),size(x_init,1),size(x_init,4))                 :: x_new,x_old
        real(8), dimension(size(x_init,1),size(env_prob,1))                                             :: y_new,y_old
        real(8), intent(out), dimension(size(x_init,1),size(x_init,2),size(x_init,1),size(x_init,4))    :: x_min
        real(8), intent(out), dimension(size(x_init,1),size(env_prob,1))                                :: y_min
        real(8), intent(out)                                                                            :: val_min

        dimS=size(env_prob,1)
        dimO=size(x_init,4)

        x_new=0.0
        val_old=0.0
        T_inst=T
        
        call norm(abs(x_init),x_old)
        y_old=y_init
        call obfun(x_old,y_old,env_prob,obs_prob,rew,sigma,mu,gamma,val_old)
        val_min=val_old
        x_min=x_old
        y_min=y_old
        countx=0
        county=0
        i=0

        do while(T_inst>Tth)
          do i=1,n
            call random_number(s)
            call norm(abs(x_old+T_inst*rndx(:,:,:,(i-1)*dimO+1+countx:i*dimO+countx)),x_new)
            y_new=abs(y_old+T_inst*rndy(:,(i-1)*dimS+1+county:i*dimS+county))
            call obfun(x_new,y_new,env_prob,obs_prob,rew,sigma,mu,gamma,val_new)
            if (exp(-(val_new-val_old)/T_inst)>s) then
              x_old=x_new
              y_old=y_new
              val_old=val_new
            endif
            if (val_new<val_min) then
              x_min=x_old
              y_min=y_old
              val_min=val_new
            endif
          enddo

          countx=countx+n*dimO
          county=county+n*dimS
          T_inst=factor*T_inst
        enddo
    
    end subroutine

    subroutine drive(T,Tth,factor,nstep,pen_step,x_init,y_init,val_init,rew,env_prob,&
        obs_prob,sigma,mu,x_gauss,y_gauss,gamma,x_min,y_min,val_min)
        real(8), intent(in)                                                                             :: T,Tth,factor
        real(8), intent(in)                                                                             :: sigma,gamma,mu,val_init
        real(8), intent(in), dimension(:,:,:,:)                                                         :: x_gauss, x_init
        real(8), intent(in), dimension(:,:,:)                                                           :: env_prob,obs_prob
        real(8), intent(in), dimension(:,:)                                                             :: rew,y_gauss,y_init
        integer                                                                                         :: countx,county
        integer                                                                                         :: nstep,pen_step
        integer                                                                                         :: i,size_gauss_x
        integer                                                                                         :: size_gauss_y
        integer                                                                                         :: indexx1,indexx2
        integer                                                                                         :: indexy1,indexy2
        real(8)                                                                                         :: sigma_inst
        real(8)                                                                                         :: mu_inst,valRes
        real(8), dimension(size(x_init,1),size(x_init,2),size(x_init,1),size(x_init,4))                 :: xRes,x_init_temp
        real(8), dimension(size(x_init,1),size(env_prob,1))                                             :: yRes,y_init_temp
        real(8), intent(out), dimension(size(x_init,1),size(x_init,2),size(x_init,1),size(x_init,4))    :: x_min
        real(8), intent(out), dimension(size(x_init,1),size(env_prob,1))                                :: y_min
        real(8), intent(out)                                                                            :: val_min

        size_gauss_x=size(x_gauss,4)/pen_step
        size_gauss_y=size(y_gauss,2)/pen_step
        countx=0
        county=0
        sigma_inst=sigma
        mu_inst=mu
        x_init_temp=x_init
        y_init_temp=y_init
        xRes=0.0
        yRes=0.0
        valRes=0.0
        val_min=-val_init
        indexx1=0
        indexx2=0
        indexy1=0
        indexy2=0

        open(unit=2,file="Trend.txt")


        do i=1,pen_step
            sigma_inst=sigma_inst*1.05
            mu_inst=mu_inst*1.05

            indexx1=(i-1)*size_gauss_x+1
            indexx2=i*size_gauss_x
            indexy1=(i-1)*size_gauss_y+1
            indexy2=i*size_gauss_y

            call sa(T,Tth,factor,nstep,x_init_temp,y_init_temp,rew,env_prob,&
            obs_prob,sigma_inst,mu_inst,x_gauss(:,:,:,indexx1:indexx2),y_gauss(:,indexy1:indexy2),gamma,xRes,yRes,valRes)

            if (valRes < -val_init) then
                x_init_temp=xRes
                y_init_temp=yRes

                x_min=x_init_temp
                y_min=y_init_temp
                val_min=valRes
            else
                x_init_temp=x_init
                y_init_temp=y_init

                x_min=x_init
                y_min=y_init
                val_min=-val_init
            endif

            write(2,*) sigma_inst, val_min

        enddo

        close(2)
   
    end subroutine
end module